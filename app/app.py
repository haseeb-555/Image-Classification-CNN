import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
from io import BytesIO

app = Flask(__name__)

# Load the pre-trained model (ensure it's in the same directory or adjust path)
model = load_model('flower_model.h5')

# Define the data augmentation configuration as during training (but for inference, we'll only use rescale)
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Rescaling for normalization (same as during training)
)

# Image size you used for training
target_size = (224, 224)

# Load the saved class labels (same as during training)
try:
    with open('class_labels.json', 'r') as f:
        class_labels = json.load(f)
except FileNotFoundError:
    class_labels = {}  # Handle the case if the file is not found
    print("Warning: 'class_labels.json' not found.")

# Reverse the dictionary to map indices to class names
index_to_class = {v: k for k, v in class_labels.items()}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to make predictions on uploaded image.
    """
    # Check if an image is part of the POST request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file is valid
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Check if the file is an image
    if file and file.filename.lower().endswith(('jpg', 'jpeg', 'png')):
        try:
            # Convert the file to a BytesIO object
            img_bytes = file.read()
            img = load_img(BytesIO(img_bytes), target_size=target_size)

            # Convert the image to a numpy array
            img_array = img_to_array(img)

            # Expand dimensions to match model input (batch size, height, width, channels)
            img_array = np.expand_dims(img_array, axis=0)

            # Apply rescaling (same as in training, using only rescale part of train_datagen)
            img_array = train_datagen.standardize(img_array)

            # Make the prediction
            predictions = model.predict(img_array)

            # Get the predicted class index (the index of the highest probability)
            predicted_class_index = np.argmax(predictions)

            # Get the class name from the index
            predicted_class = index_to_class.get(predicted_class_index, "Unknown class")

            # Return the result as JSON
            return jsonify({'prediction': predicted_class})

        except Exception as e:
            return jsonify({'error': f"Error during prediction: {str(e)}"})

    else:
        return jsonify({'error': 'Invalid file format. Please upload a jpg, jpeg, or png file.'})

if __name__ == '__main__':
    # Set the host to '0.0.0.0' to make it accessible externally
    app.run(debug=True, host='0.0.0.0', port=5000)
