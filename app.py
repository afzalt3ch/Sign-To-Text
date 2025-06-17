import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import torchvision.transforms as transforms
import os

# Initialize Flask app
app = Flask(__name__)

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the model weights
MODEL_PATH = r"C:\Users\afzal\OneDrive\Desktop\Projects\SignToText\model\efficientnet_model.pth"

# Load the model and weights at the start
print("Loading model...")

try:
    # Load the pretrained EfficientNet model
    model = EfficientNet.from_pretrained('efficientnet-b0')
    
    # Modify the output layer to match the number of classes (29)
    model._fc = nn.Linear(model._fc.in_features, 29)  
    model = model.to(device)  # Move model to GPU or CPU

    # Check if the model weights exist and load them
    if os.path.exists(MODEL_PATH):  
        state_dict = torch.load(MODEL_PATH, map_location=device)
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}  # Remove prefix from state_dict keys
        model.load_state_dict(state_dict)  # Load the weights into the model
        model.eval()  # Set the model to evaluation mode
        print("Model loaded successfully!")
    else:
        print(f"Model weights not found at {MODEL_PATH}")
        model = None  # In case model loading fails
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define image preprocessing function
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to 128x128
        transforms.ToTensor(),  # Convert image to tensor
    ])
    
    # Apply transformations and add batch dimension
    return transform(image).unsqueeze(0)  # Add batch dimension

# Class label mapping (29 classes for ASL)
CLASS_LABELS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        image_bytes = file.read()
    
        # Open image, ensure it's in RGB format
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")  # Convert to RGB to ensure compatibility

        # Preprocess image (resize, convert to tensor, normalize)
        input_tensor = preprocess_image(image).to(device)

        # Make prediction with no gradient computation
        with torch.no_grad():
            outputs = model(input_tensor)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
            class_idx = predicted.item()

        # Map the class index to the label
        class_label = CLASS_LABELS.get(class_idx, "Unknown")  # Map class index to label

        print(f"Prediction made: {class_label}")
        
        # Return the predicted class label in the response
        return jsonify({"predicted_class": class_label, "class_idx": class_idx})
        
    except FileNotFoundError:
        print("File not found error.")
        return jsonify({"error": "File not found."}), 400

    except IOError:
        print("IO error occurred while processing the image.")
        return jsonify({"error": "Error opening image."}), 400

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": f"Unexpected error during prediction: {str(e)}"}), 500
    # Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  # Set debug=False in production
