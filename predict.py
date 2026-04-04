import os
import sys
import json
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# ─── Configuration ───
IMG_SIZE   = 128
MODEL_PATH = 'plant_disease_model.h5'
CLASS_JSON = 'class_indices.json'

# ─── Disease Advice Dictionary ───
DISEASE_INFO = {
    'Tomato_healthy': {
        'status' : '✅ Healthy',
        'advice' : 'Your plant looks healthy! Continue regular watering and care.',
        'color'  : 'green'
    },
    'Tomato_Early_blight': {
        'status' : '⚠️ Early Blight Detected',
        'advice' : 'Apply copper-based fungicide. Remove affected leaves. Improve air circulation.',
        'color'  : 'orange'
    },
    'Tomato_Late_blight': {
        'status' : '🚨 Late Blight Detected',
        'advice' : 'Serious disease! Apply fungicide immediately. Remove and destroy infected plants.',
        'color'  : 'red'
    }
}


def load_resources():
    """Load the trained model and class index mapping."""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("   Please train the model first using the Jupyter Notebook.")
        sys.exit(1)

    if not os.path.exists(CLASS_JSON):
        print(f"❌ Class index file not found: {CLASS_JSON}")
        sys.exit(1)

    model = load_model(MODEL_PATH)
    with open(CLASS_JSON, 'r') as f:
        class_indices = json.load(f)

    print("✅ Model and class indices loaded!")
    return model, class_indices


def preprocess_image(image_path):
    """Load and preprocess a single image for prediction."""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    # Load image using Keras utility (handles various formats)
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)       # Shape: (128, 128, 3)
    img_array = img_array / 255.0       # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 3)
    return img_array


def predict(image_path, model, class_indices):
    """Run prediction on the given image."""
    # Preprocess
    img_array = preprocess_image(image_path)

    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_idx   = np.argmax(predictions[0])
    confidence      = predictions[0][predicted_idx] * 100
    predicted_class = class_indices[str(predicted_idx)]

    # All class probabilities
    all_probs = {class_indices[str(i)]: predictions[0][i] * 100
                 for i in range(len(class_indices))}

    return predicted_class, confidence, all_probs


def display_result(image_path, predicted_class, confidence, all_probs):
    """Display the image with prediction result."""
    info = DISEASE_INFO.get(predicted_class, {
        'status': predicted_class,
        'advice': 'Consult an agricultural expert.',
        'color' : 'blue'
    })

    # Read image with OpenCV
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#f8f8f8')

    # ── Left: Image with prediction ──
    ax1.imshow(img)
    ax1.set_title(
        f"{info['status']}\nConfidence: {confidence:.1f}%",
        fontsize=13, fontweight='bold', color=info['color'], pad=12
    )
    ax1.axis('off')

    # ── Right: Probability bar chart ──
    classes = list(all_probs.keys())
    probs   = list(all_probs.values())
    colors  = ['#e74c3c' if c == predicted_class else '#3498db' for c in classes]
    short_names = [c.replace('Tomato_', '').replace('_', ' ') for c in classes]

    bars = ax2.barh(short_names, probs, color=colors, edgecolor='white', height=0.5)
    ax2.set_xlabel('Confidence (%)', fontsize=11)
    ax2.set_title('Class Probabilities', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.grid(axis='x', alpha=0.3)
    for bar, prob in zip(bars, probs):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f'{prob:.1f}%', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=100, bbox_inches='tight')
    plt.show()

    # ── Print to console ──
    print("\n" + "="*50)
    print(f"  PLANT DISEASE DETECTION RESULT")
    print("="*50)
    print(f"  Image     : {os.path.basename(image_path)}")
    print(f"  Predicted : {predicted_class}")
    print(f"  Confidence: {confidence:.2f}%")
    print(f"\n  💡 Advice : {info['advice']}")
    print("="*50)
    print("\nAll probabilities:")
    for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        bar = '█' * int(prob / 5)
        print(f"  {cls:<30} {prob:5.1f}% {bar}")


def main():
    parser = argparse.ArgumentParser(
        description='Predict plant disease from a leaf image.'
    )
    parser.add_argument(
        '--image', type=str, required=True,
        help='Path to the leaf image (e.g., dataset/test/Tomato_healthy/img.jpg)'
    )
    args = parser.parse_args()

    print("\n🌿 Plant Disease Detection System")
    print("-" * 40)

    # Load model
    model, class_indices = load_resources()

    # Predict
    predicted_class, confidence, all_probs = predict(args.image, model, class_indices)

    # Show result
    display_result(args.image, predicted_class, confidence, all_probs)


if __name__ == '__main__':
    main()
