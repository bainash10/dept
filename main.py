from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from PIL import Image
import os
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

# Store model paths instead of loading them immediately
MODEL_PATHS = {
    'lenet5': '50lenet5.keras',
    'ran2dev': '50ran2dev.keras'
}

# Lazy model loading (to avoid memory overload)
loaded_models = {}

# Class labels for Ranjana script
class_labels = [
    'अ', 'आ', 'अ:', 'ऐ', 'अं', 'औ', 'ब', 'भ', 'च', 'छ', 'ड',
    'द', 'ध', 'ढ', 'ए', '८', '५', '५', 'ग', 'घ', 'ज्ञ',
    'ह', 'इ', 'ई', 'ज', 'झ', 'क', 'ख', 'क्ष', 'ल', 'लृ', 'luu',
    'म', 'न,', '९', 'ण', 'न', 'ञ', 'ओ', '१', 'प', 'फ',
    'र', 'ऋ', 'rii', 'ष', 'स', '७', 'श', '६', 'ट', 'ठ',
    '३', 'त्र', 'त', 'थ', '२', 'उ', 'ऊ', 'व', 'य', '०',
]

IMAGE_SIZES = {
    'ran2dev': (64, 64),
    'lenet5': (32, 32)
}

def load_model_lazy(model_name):
    """Load model only when needed to avoid memory issues."""
    if model_name not in loaded_models:
        try:
            loaded_models[model_name] = tf.keras.models.load_model(MODEL_PATHS[model_name])
        except Exception as e:
            return str(e)  # Return error message if model fails to load
    return loaded_models[model_name]

def prepare_image(image, model_name):
    """Preprocess image: Resize, convert to grayscale, and apply Otsu thresholding."""
    image = Image.open(image).convert("RGB")  
    image = np.array(image, dtype=np.uint8)

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = cv2.resize(image, (67, 78), interpolation=cv2.INTER_AREA)
    _, otsu_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    target_size = IMAGE_SIZES[model_name]
    otsu_image = cv2.resize(otsu_image, target_size, interpolation=cv2.INTER_AREA)
    otsu_image = otsu_image / 255.0
    otsu_image = np.reshape(otsu_image, (1, target_size[1], target_size[0], 1))

    return otsu_image

def segment_characters(image, model_name):
    """Segment characters from the word image and prepare them for prediction."""
    image = Image.open(image).convert("L")
    image = np.array(image, dtype=np.uint8)
    
    # Apply thresholding for segmentation
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    character_images = []
    bounding_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        char_image = image[y:y+h, x:x+w]
        char_image = cv2.resize(char_image, IMAGE_SIZES[model_name])
        char_image = char_image.astype('float32') / 255.0
        char_image = np.expand_dims(char_image, axis=(0, -1))
        character_images.append(char_image)
        bounding_boxes.append((x, y, w, h))
    
    # Sort characters left to right
    sorted_characters = sorted(zip(character_images, bounding_boxes), key=lambda b: b[1][0])
    return [char[0] for char in sorted_characters]

def predict_characters(image, model_name):
    """Predict the word using the chosen model."""
    character_images = segment_characters(image, model_name)
    model = load_model_lazy(model_name)
    
    if isinstance(model, str):  # Model failed to load
        return model

    combined_prediction = ""

    for char_image in character_images:
        pred = model.predict(char_image)
        predicted_class_index = np.argmax(pred, axis=-1)[0]
        predicted_class_label = class_labels[predicted_class_index]
        combined_prediction += predicted_class_label

    return combined_prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict a single character using the chosen model."""
    if 'image' not in request.files or 'model' not in request.form:
        return jsonify({"error": "No file or model selected"}), 400
    
    image_file = request.files['image']
    model_name = request.form['model']
    
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if model_name not in MODEL_PATHS:
        return jsonify({"error": "Invalid model selected"}), 400

    model = load_model_lazy(model_name)
    if isinstance(model, str):  # Model failed to load
        return jsonify({"error": f"Model loading failed: {model}"}), 500

    try:
        image = prepare_image(image_file, model_name)
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_labels[predicted_class]
        
        return jsonify({"prediction": predicted_label})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route('/predict-word', methods=['POST'])
def predict_word():
    """Predict a word using the chosen model."""
    if 'image' not in request.files or 'model' not in request.form:
        return jsonify({"error": "No file or model selected"}), 400
    
    image_file = request.files['image']
    model_name = request.form['model']
    
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if model_name not in MODEL_PATHS:
        return jsonify({"error": "Invalid model selected"}), 400

    try:
        predicted_text = predict_characters(image_file, model_name)
        
        if isinstance(predicted_text, str) and "error" in predicted_text.lower():
            return jsonify({"error": predicted_text}), 500

        return jsonify({"prediction": predicted_text})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
