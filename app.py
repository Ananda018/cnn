from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image


# Initialize app and enable CORS
app = Flask(__name__)
CORS(app)

# Load model
model = load_model("my_final_model.h5")

# Your class names
class_names = ['Normal OCT', 'Sharp Peaked PED']

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]

    return jsonify({'class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
