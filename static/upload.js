console.log("upload.js script loaded."); // Add this line

document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM fully loaded and parsed."); // Add this line
    const uploadBox = document.getElementById('uploadBox');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const removeBtn = document.getElementById('removeBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultsCard = document.getElementById('resultsCard');
    const healthScore = document.getElementById('healthScore');
    const scoreNumber = document.getElementById('scoreNumber');
    const conditionsList = document.getElementById('conditionsList');
    const recommendations = document.getElementById('recommendations');
    const uploadContent = uploadBox.querySelector('.upload-content'); // Get upload content div

    // Chatbot elements
    const chatInput = document.getElementById('chatInput');
    const sendButton = document.getElementById('sendButton');
    const chatMessages = document.getElementById('chatMessages');
    const errorMessageDiv = document.getElementById('err'); // Error message div

    // Store chat history
    let conversationHistory = [];

    // --- Event Listeners ---
    if (uploadBox) {
        console.log("Upload box found, adding event listeners."); // Add this line
        // Handle drag and drop
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadBox.addEventListener(eventName, preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadBox.addEventListener(eventName, () => uploadBox.classList.add('drag-over'));
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadBox.addEventListener(eventName, () => uploadBox.classList.remove('drag-over'));
        });

        uploadBox.addEventListener('drop', handleDrop);
        // Make the entire box clickable if fileInput exists
        if (fileInput) {
            uploadBox.addEventListener('click', (e) => {
                // Prevent triggering click if remove button is clicked
                if (e.target !== removeBtn) {
                    fileInput.click();
                }
            });
        }
    } else {
        console.error("Upload box element not found!"); // Add this line
    }

    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    } else {
        console.error("File input element not found!"); // Add this line
    }
    if (removeBtn) {
        removeBtn.addEventListener('click', removeImage);
    }
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', analyzeFur);
    }
    if (sendButton) {
        sendButton.addEventListener('click', sendMessage);
    }
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }

    // --- Functions ---
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFileSelect(e) {
        const files = e.target.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        console.log("handleFiles called with", files.length, "file(s)."); // Add this line
        if (files.length > 0) {
            const file = files[0];
            // Basic validation (can be enhanced)
            if (file.type.startsWith('image/')) {
                console.log("File is an image, proceeding with preview."); // Add this line
                const reader = new FileReader();
                reader.onload = (e) => {
                    if (imagePreview) imagePreview.src = e.target.result;
                    if (previewContainer) previewContainer.classList.add('active');
                    if (uploadContent) uploadContent.style.display = 'none'; // Hide upload text/icon
                    if (analyzeBtn) {
                        console.log("Enabling analyze button."); // Add this line
                        analyzeBtn.disabled = false;
                    } else {
                        console.error("Analyze button not found when trying to enable!"); // Add this line
                    }
                    if (resultsCard) resultsCard.classList.remove('active'); // Hide previous results
                    clearErrorMessage(); // Clear any previous errors
                };
                reader.onerror = (e) => { // Add error handling for reader
                    console.error("FileReader error:", e);
                    showErrorMessage('Error reading the selected file.');
                };
                reader.readAsDataURL(file);
            } else {
                console.warn("Invalid file type selected:", file.type); // Add this line
                showErrorMessage('Please upload an image file (JPG, PNG, JPEG).');
                removeImage(); // Reset if invalid file type
            }
        }
    }

    function removeImage() {
        console.log("removeImage called."); // Add this line
        if (imagePreview) imagePreview.src = '#'; // Use '#' or ''
        if (previewContainer) previewContainer.classList.remove('active');
        if (uploadContent) uploadContent.style.display = 'block'; // Show upload text/icon
        if (analyzeBtn) analyzeBtn.disabled = true;
        if (resultsCard) resultsCard.classList.remove('active'); // Hide results card
        if (fileInput) fileInput.value = ''; // Clear the file input
        clearErrorMessage();
    }

    function analyzeFur() {
        console.log("analyzeFur called."); // Add this line
        if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
            showErrorMessage('Please select an image file first.');
            console.warn("analyzeFur called without a file selected."); // Add this line
            return;
        }

        if (analyzeBtn) {
            analyzeBtn.classList.add('analyzing');
            analyzeBtn.disabled = true;
            console.log("Analyze button disabled, starting analysis."); // Add this line
        }
        clearErrorMessage();

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        console.log("Sending fetch request to /predict"); // Add this line

        fetch('/predict', { // Ensure this matches the Flask route
            method: 'POST',
            body: formData
        })
        .then(response => {
            console.log("Received response from /predict with status:", response.status); // Add this line
            if (!response.ok) {
                // Try to get error message from response body if possible
                return response.json().then(errData => {
                    console.error("Server returned error:", errData); // Add this line
                    throw new Error(errData.error || `Server error: ${response.statusText}`);
                }).catch((jsonError) => { // Catch potential JSON parsing error
                    console.error("Failed to parse error response as JSON:", jsonError); // Add this line
                    // Fallback if response body isn't JSON or doesn't have error field
                    throw new Error(`Server error: ${response.status} ${response.statusText}`);
                });
            }
            return response.json();
        })
        .then(data => {
            console.log("Prediction successful:", data); // Add this line
            displayResults(data);
            // Add a message to the chat about the results
            addAssistantMessage(`• I've analyzed your pet's fur image.
• The analysis suggests: ${data.prediction || 'potential issues'} with ${data.confidence || 'N/A'}% confidence.
• Ask me any questions about this condition or general fur care.`);
        })
        .catch(error => {
            console.error('Analysis Fetch Error:', error); // Modify log prefix
            showErrorMessage(`Analysis failed: ${error.message}. Please try again.`);
            // Optionally hide results card on error
            if (resultsCard) resultsCard.classList.remove('active');
        })
        .finally(() => {
            if (analyzeBtn) {
                analyzeBtn.classList.remove('analyzing');
                console.log("Analysis finished, removing analyzing state."); // Add this line
                // Keep button disabled until a new image is selected or current one removed?
                // analyzeBtn.disabled = false; // Re-enable immediately
            }
        });
    }

    function displayResults(data) {
        if (!resultsCard || !healthScore || !scoreNumber || !conditionsList || !recommendations) {
            console.error("Result display elements not found.");
            return;
        }

        resultsCard.classList.add('active');

        const confidenceScore = data.confidence ? parseFloat(data.confidence) : 0;
        const score = Math.round(confidenceScore);

        // Animate score bar
        setTimeout(() => {
            healthScore.style.width = `${score}%`;
        }, 100); // Small delay for transition effect

        scoreNumber.textContent = `${score}%`;

        // Display conditions
        conditionsList.innerHTML = ''; // Clear previous conditions
        const conditionItem = document.createElement('li');
        conditionItem.textContent = `Detected condition: ${data.prediction || 'Unknown'}`;
        conditionsList.appendChild(conditionItem);

        if (data.prediction && data.prediction !== 'Healthy') {
            const vetItem = document.createElement('li');
            vetItem.textContent = 'Veterinary consultation recommended for confirmation.';
            conditionsList.appendChild(vetItem);
        } else if (data.prediction === 'Healthy') {
            const healthyItem = document.createElement('li');
            healthyItem.textContent = 'Skin appears healthy based on the image.';
            conditionsList.appendChild(healthyItem);
        }

        // Display recommendations (customize based on prediction)
        let recommendationText = '';
        switch (data.prediction) {
            case 'Healthy':
                recommendationText = `• Based on our analysis, your pet's fur appears to be in good condition (${score}% confidence).\n• Continue regular grooming and monitor for any changes.\n• Maintain a balanced diet and provide fresh water.`;
                break;
            case 'Dermatitis':
                recommendationText = `• Signs consistent with Dermatitis detected (${score}% confidence). This involves skin inflammation, often causing itching and redness.\n• Potential Remedies: Soothing baths with oatmeal-based shampoo, keeping the area clean and dry.\n• Precautions: Identify and avoid potential irritants (e.g., new foods, cleaning products, plants).\n• Consult a vet for diagnosis and specific treatment (e.g., medicated shampoos, allergy management, medication).`;
                break;
            case 'Fungal_infections':
            case 'ringworm': // Ringworm is fungal
                recommendationText = `• Signs consistent with a Fungal Infection (${data.prediction}, ${score}% confidence) detected. Often causes itchy, circular patches and hair loss.\n• Potential Remedies: Keep the area clean and dry. Antifungal shampoos may help manage symptoms.\n• Precautions: Fungal infections like ringworm can be contagious to other pets and humans. Wash bedding frequently and limit contact until treated.\n• Consult a vet for diagnosis (e.g., fungal culture) and appropriate treatment (e.g., topical or oral antifungal medications).`;
                break;
            case 'Hypersensitivity':
                recommendationText = `• Signs consistent with Hypersensitivity (Allergic Reaction) detected (${score}% confidence). This causes itchy, irritated skin.\n• Potential Remedies: Identify and eliminate the allergen if possible (e.g., food trial, environmental changes). Soothing shampoos can provide temporary relief.\n• Precautions: Monitor for severe reactions. Keep a record of potential triggers.\n• Consult a vet for diagnosis, allergy testing, and management strategies (e.g., diet changes, antihistamines, immunotherapy).`;
                break;
            case 'demodicosis': // Mange caused by Demodex mites
                recommendationText = `• Signs consistent with Demodicosis (Mange) detected (${score}% confidence). Caused by Demodex mites, often leading to hair loss and sometimes skin thickening.\n• Potential Remedies: Generally requires specific veterinary treatment. Boosting the immune system through good nutrition may help.\n• Precautions: While less contagious than Sarcoptic mange, follow vet advice on handling and hygiene. Usually related to an underlying immune issue.\n• Consult a vet for diagnosis (e.g., skin scraping) and specific treatment (e.g., miticidal medication, addressing underlying conditions).`;
                break;
            default:
                recommendationText = `• Analysis complete. Predicted condition: ${data.prediction || 'Unknown'} (${score}% confidence).\n• General Care: Keep the affected area clean and prevent excessive licking or scratching.\n• Since the condition is less common or unclear from the image, it's crucial to get a professional opinion.\n• Consult a veterinarian for a definitive diagnosis and appropriate care plan.`;
        }
        // Replace newline characters with <br> tags for HTML display
        recommendations.innerHTML = recommendationText.replace(/\n/g, '<br>');

        resultsCard.scrollIntoView({ behavior: 'smooth' });
    }

    // --- Chatbot Functions ---
    function sendMessage() {
        if (!chatInput || !chatMessages) return;
        const message = chatInput.value.trim();
        if (!message) return;

        addUserMessage(message);
        chatInput.value = '';
        clearErrorMessage(); // Clear errors when user sends a message

        // Add thinking indicator? (Optional)

        fetch('/chatbot', { // Ensure this matches the Flask route
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                history: conversationHistory // Send history
            }),
        })
        .then(response => {
            if (!response.ok) {
                // Try to get error message from response body if possible
                return response.json().then(errData => {
                    throw new Error(errData.error || `Server error: ${response.statusText}`);
                }).catch(() => {
                    // Fallback if response body isn't JSON or doesn't have error field
                    throw new Error(`Chatbot error: ${response.status} ${response.statusText}`);
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.response) {
                addAssistantMessage(data.response);
            } else {
                throw new Error('Invalid response format from chatbot');
            }
        })
        .catch(error => {
            console.error('Chatbot Error:', error);
            addAssistantMessage(`• I apologize, I encountered an error: ${error.message}.
• Please try again later or consult a vet for urgent help.`);
        });
    }

    function addUserMessage(text) {
        if (!chatMessages) return;
        const messageElement = createMessageElement(text, 'user');
        chatMessages.appendChild(messageElement);
        scrollToBottom();
        // Add to history
        conversationHistory.push({ role: "user", content: text });
        // Limit history size (e.g., keep last 10 messages total)
        if (conversationHistory.length > 10) {
            conversationHistory = conversationHistory.slice(-10);
        }
    }

    function addAssistantMessage(text) {
        if (!chatMessages) return;
        const messageElement = createMessageElement(text, 'assistant');
        chatMessages.appendChild(messageElement);
        scrollToBottom();
        // Add to history
        conversationHistory.push({ role: "assistant", content: text });
        // Limit history size
        if (conversationHistory.length > 10) {
            conversationHistory = conversationHistory.slice(-10);
        }
    }

    function createMessageElement(text, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        // Basic formatting for newlines and bullet points
        const formattedText = text
            .replace(/•/g, '• ') // Ensure space after bullet
            .replace(/\n/g, '<br>'); // Convert newlines to <br>

        messageContent.innerHTML = formattedText; // Use innerHTML to render <br>

        messageDiv.appendChild(messageContent);
        return messageDiv;
    }

    function scrollToBottom() {
        if (chatMessages) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }

    function showErrorMessage(message) {
        console.log("Showing error message:", message); // Add this line
        if (errorMessageDiv) {
            errorMessageDiv.textContent = message;
            errorMessageDiv.style.color = 'red'; // Make error visible
            errorMessageDiv.style.marginTop = '1rem';
        }
        console.error("Error Displayed:", message); // Also log to console
    }

    function clearErrorMessage() {
        console.log("Clearing error message."); // Add this line
        if (errorMessageDiv) {
            errorMessageDiv.textContent = '';
            errorMessageDiv.style.marginTop = '0';
        }
    }

    // Initial setup if needed (e.g., add initial assistant message)
    // addAssistantMessage("• Hello! I'm PawHelp. Upload a fur image above and ask me questions.");

}); // End DOMContentLoaded
