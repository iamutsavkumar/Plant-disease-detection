"""
app.py — Flask Web Application for Plant Disease Detection
==========================================================
Run: python app.py
Then open your browser at: http://127.0.0.1:5000
"""

import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# ─── App Configuration ───
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# ─── Model & Config ───
IMG_SIZE   = 128
MODEL_PATH = 'plant_disease_model.h5'
CLASS_JSON = 'class_indices.json'

# Disease info for display
DISEASE_INFO = {
    'Tomato_healthy': {
        'label'   : 'Healthy Tomato',
        'status'  : 'Healthy',
        'badge'   : 'success',
        'icon'    : '✅',
        'advice'  : 'Your plant is healthy! Continue with regular watering (once every 2-3 days), ensure good sunlight, and monitor for any future changes.',
        'severity': 'None'
    },
    'Tomato_Early_blight': {
        'label'   : 'Early Blight',
        'status'  : 'Diseased',
        'badge'   : 'warning',
        'icon'    : '⚠️',
        'advice'  : 'Apply copper-based or mancozeb fungicide. Remove affected lower leaves. Avoid overhead watering. Improve plant spacing for better airflow.',
        'severity': 'Moderate'
    },
    'Tomato_Late_blight': {
        'label'   : 'Late Blight',
        'status'  : 'Severely Diseased',
        'badge'   : 'danger',
        'icon'    : '🚨',
        'advice'  : 'Act immediately! Apply systemic fungicide (chlorothalonil or metalaxyl). Remove and destroy all infected plant material. Avoid wet conditions.',
        'severity': 'High'
    }
}

# ─── Load Model at Startup ───
print("Loading model...")
if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_JSON):
    model = load_model(MODEL_PATH)
    with open(CLASS_JSON, 'r') as f:
        class_indices = json.load(f)
    print("✅ Model loaded successfully!")
else:
    model = None
    class_indices = {}
    print("⚠️  Model not found. Train the model first using the Jupyter notebook.")

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """Check if uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_path):
    """
    Preprocess and predict the disease for a leaf image.
    Returns: (class_name, confidence, all_probabilities)
    """
    # Load and preprocess
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array, verbose=0)
    pred_idx   = np.argmax(preds[0])
    confidence = float(preds[0][pred_idx]) * 100
    pred_class = class_indices[str(pred_idx)]

    all_probs = {
        class_indices[str(i)]: round(float(preds[0][i]) * 100, 2)
        for i in range(len(class_indices))
    }

    return pred_class, confidence, all_probs


# ─── Routes ───

@app.route('/')
def index():
    """Home page — upload form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction."""
    if model is None:
        return jsonify({'error': 'Model not loaded. Train the model first.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use JPG, PNG, or GIF.'}), 400

    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Predict
    pred_class, confidence, all_probs = predict_image(filepath)
    info = DISEASE_INFO.get(pred_class, {
        'label'   : pred_class,
        'status'  : 'Unknown',
        'badge'   : 'secondary',
        'icon'    : '❓',
        'advice'  : 'Please consult an agricultural expert.',
        'severity': 'Unknown'
    })

    return jsonify({
        'success'       : True,
        'image_path'    : f'uploads/{filename}',
        'predicted_class': pred_class,
        'label'         : info['label'],
        'status'        : info['status'],
        'badge'         : info['badge'],
        'icon'          : info['icon'],
        'advice'        : info['advice'],
        'severity'      : info['severity'],
        'confidence'    : round(confidence, 2),
        'all_probs'     : all_probs
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
