import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import numpy as np

app = Flask(__name__)

# Load your Keras model and define class names
model1 = keras.models.load_model('model/model1.h5')
class_names1 = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# Define the upload folder for images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to preprocess the image and make predictions using the first model
def process_image(image_path):
    img = Image.open(image_path)
    img1 = img.resize((256, 256))
    img1 = img1.convert("RGB")
    img1 = np.array(img1)
    img1 = img1 / 255.0
    predictions1 = model1.predict(np.expand_dims(img1, axis=0))
    predicted_class1 = class_names1[np.argmax(predictions1, axis=1)[0]]
    return predicted_class1

@app.route('/')
def index():
    return render_template('frontpage.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']

    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded image to the static/uploads folder
        image_path = os.path.join("static/uploads", file.filename)
        file.save(image_path)

        # Process the image and make predictions
        predicted_class1 = process_image(image_path)

        # Provide the predicted class and image URL to the result page
        result = {
            'predicted_class1': predicted_class1,
            'image_url': file.filename  # Just the filename, not the full path
        }
        return render_template('result.html', result=result)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=False, port=900)
