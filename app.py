import os
import numpy as np
import torch
import torchvision
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'supersecretkey'

latest_prediction = {
    'disease': None,
    'confidence': None,
    'timestamp': None
}

# Gemini API setup
genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # Changed from GEMINI_API_KEY

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Load Model ---
MODEL_PATH_PT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kaggle_output1', 'best_model.pth')
NUM_CLASSES_PT = 6
CLASS_LABELS_PT = ['Dermatitis', 'Fungal_infections', 'Healthy', 'Hypersensitivity', 'demodicosis', 'ringworm']
model_pt = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pytorch_model(model_path, num_classes):
    try:
        print(f"Loading PyTorch model from {model_path}...")
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("PyTorch Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        return None

model_pt = load_pytorch_model(MODEL_PATH_PT, NUM_CLASSES_PT)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_pt(image_path):
    try:
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = Image.open(image_path).convert('RGB')
        img_t = preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0)
        return batch_t.to(device)
    except Exception as e:
        print(f"Error preprocessing image {image_path} for PyTorch: {e}")
        return None

# --- Chatbot System Prompt ---
SYSTEM_PROMPT = """
You are a mini chatbot assistant named PawHelp that specializes in dog skin diseases. Your responses must be:
1. STRUCTURED - Always use bullet points for all responses
2. BRIEF - Keep all responses concise with 3-5 bullet points
3. SIMPLE - Use clear, non-technical language
4. FOCUSED - Address exactly what was asked

Format ALL responses with bullet points (•) at the start of each line.

You know about these skin conditions:
• Dermatitis: Skin inflammation causing itching and redness
• Fungal Infections: Itchy patches with hair loss
• Healthy Skin: Normal skin with no issues
• Hypersensitivity: Allergic reaction causing skin irritation
• Demodicosis: Mites in hair follicles causing hair loss
• Ringworm: Circular patches of hair loss (actually a fungal infection)

Include "• Consult a vet for proper diagnosis and treatment" as the final point in disease-related responses.
"""

generation_config = {
    "temperature": 0.2,
    "top_p": 0.85,
    "top_k": 20,
    "max_output_tokens": 250,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    safety_settings=safety_settings
)

def get_advanced_chatbot_response(user_message, conversation_history=None):
    if conversation_history is None:
        conversation_history = []

    if not os.getenv("GOOGLE_API_KEY"): # Changed from GEMINI_API_KEY
        return "• I'm sorry, the chatbot is not properly configured\n• Please make sure the GOOGLE_API_KEY is set in your .env file"

    try:
        chat = gemini_model.start_chat(history=[])
        chat.send_message(SYSTEM_PROMPT)

        if latest_prediction['disease']:
            chat.send_message(
                f"The user uploaded a dog's skin image, and the system detected "
                f"{latest_prediction['disease']} with {latest_prediction['confidence']:.2f}% confidence."
            )

        for msg in conversation_history[-5:]:
            role = msg.get("role")
            content = msg.get("content")
            if role == "user":
                chat.send_message(content)
            elif role == "assistant":
                chat.send_message(f"The AI assistant previously responded: {content}")

        response = chat.send_message(user_message)
        text = response.text.strip()

        if not text.startswith("•"):
            lines = text.split('\n')
            text = '\n'.join(f"• {line.strip()}" if line.strip() and not line.strip().startswith("•") else line for line in lines)

        return text
    except Exception as e:
        print(f"Chatbot error: {e}")
        return "• I apologize, the chatbot service is currently unavailable\n• Please consult a vet for urgent help"

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/page2')
def page2():
    return render_template('page2.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model_pt is None:
        return jsonify({'error': 'Model not loaded. Cannot perform prediction.'}), 500

    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file part in the request.'}), 400
    if file.filename == '':
        return jsonify({'error': 'No file selected for upload.'}), 400
    if not allowed_file(file.filename):
        allowed_ext_str = ", ".join(ALLOWED_EXTENSIONS)
        return jsonify({'error': f'Invalid file type. Allowed types: {allowed_ext_str}'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(filepath)
        processed_image = preprocess_image_pt(filepath)
        if processed_image is None:
            return jsonify({'error': 'Image preprocessing failed. The image might be corrupted or in an unsupported format.'}), 400

        with torch.no_grad():
            outputs = model_pt(processed_image)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, idx = torch.max(probs, 1)

        label = CLASS_LABELS_PT[idx.item()]
        confidence_score = confidence.item() * 100

        global latest_prediction
        import datetime
        latest_prediction = {
            'disease': label,
            'confidence': confidence_score,
            'timestamp': datetime.datetime.now()
        }

        return jsonify({
            'prediction': label,
            'confidence': f"{confidence_score:.2f}",
            'image_file': filename
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'An unexpected error occurred during prediction: {str(e)}'}), 500

@app.route('/chatbot', methods=['POST'])
def chatbot():
    if not request.is_json:
        return jsonify({'error': 'Invalid request format'}), 400

    data = request.get_json()
    user_message = data.get('message', '')
    conversation_history = data.get('history', [])

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    response = get_advanced_chatbot_response(user_message, conversation_history)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
