from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Initialize app and enable CORS
app = Flask(__name__)
CORS(app)

# Define model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "my_final_model.keras")

# Debugging: Check if model file exists
if not os.path.exists(MODEL_PATH):
    print("❌ Model file not found!")
    print("📂 Current directory contents:", os.listdir(os.path.dirname(__file__)))
    raise FileNotFoundError(f"Could not find the model at {MODEL_PATH}")

# Load model
print("✅ Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# Your class names
class_names = ['Normal OCT', 'Sharp Peaked PED']

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    try:
        img = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return jsonify({'error': 'Invalid image file'}), 400

    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]

    return jsonify({'class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Allow Render to bind port
