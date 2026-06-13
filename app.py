import os
import json
import io
import base64
import numpy as np
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # Maks 16 MB
app.config['UPLOAD_FOLDER'] = 'static/uploads'

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}
IMG_SIZE = 224

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def load_artifacts():
    model_path = 'uno_mobilenetv2_final.h5'
    class_path = 'class_names.json'

    if not os.path.exists(model_path):
        print(f"⚠️  Model tidak ditemukan: {model_path}")
        print("   Jalankan train.py terlebih dahulu untuk melatih model.")
        return None, []

    if not os.path.exists(class_path):
        print(f"⚠️  class_names.json tidak ditemukan.")
        return None, []

    print(f"📦 Memuat model dari {model_path} ...")
    model = load_model(model_path)

    with open(class_path, 'r') as f:
        class_names = json.load(f)

    print(f"✅ Model dimuat. Kelas: {len(class_names)}")
    return model, class_names

MODEL, CLASS_NAMES = load_artifacts()

def allowed_file(filename: str) -> bool:
    """Memeriksa apakah ekstensi file diizinkan."""
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert('RGB')
    TARGET = 224
    w, h = image.size
    scale = TARGET / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    padded = Image.new('RGB', (TARGET, TARGET), (0, 0, 0))
    pad_x = (TARGET - new_w) // 2
    pad_y = (TARGET - new_h) // 2
    padded.paste(resized, (pad_x, pad_y))
    arr = img_to_array(padded) / 255.0
    return np.expand_dims(arr, axis=0)

def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=85)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def format_class_display(class_name: str) -> str:
    return class_name.replace('_', ' ').title()


def get_color_from_class(class_name: str) -> str:
    name_lower = class_name.lower()
    if name_lower.startswith('red'):
        return '#e74c3c'
    elif name_lower.startswith('blue'):
        return '#2980b9'
    elif name_lower.startswith('green'):
        return '#27ae60'
    elif name_lower.startswith('yellow'):
        return '#f39c12'
    else:   # wild
        return '#8e44ad'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({
            'error': 'Model belum dimuat. Jalankan train.py terlebih dahulu.'
        }), 503

    image = None

    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Tidak ada file yang dipilih.'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': f'Tipe file tidak didukung. Gunakan: {ALLOWED_EXTENSIONS}'}), 400
        try:
            image = Image.open(io.BytesIO(file.read()))
        except Exception as e:
            return jsonify({'error': f'Gagal membaca gambar: {str(e)}'}), 400

    elif request.is_json and 'image' in request.json:
        try:
            img_data = base64.b64decode(request.json['image'])
            image = Image.open(io.BytesIO(img_data))
        except Exception as e:
            return jsonify({'error': f'Gagal decode base64: {str(e)}'}), 400

    else:
        return jsonify({'error': 'Tidak ada gambar dalam request.'}), 400

    try:
        tensor = preprocess_image(image)
        predictions = MODEL.predict(tensor, verbose=0)[0]

        top_idx  = int(np.argmax(predictions))
        confidence = float(predictions[top_idx]) * 100.0

        # Top-3 prediksi untuk tampilan informatif
        top3_indices = np.argsort(predictions)[::-1][:3]
        top3 = [
            {
                'class': CLASS_NAMES[i],
                'display': format_class_display(CLASS_NAMES[i]),
                'confidence': round(float(predictions[i]) * 100, 2)
            }
            for i in top3_indices
        ]

        predicted_class = CLASS_NAMES[top_idx]

        return jsonify({
            'class': predicted_class,
            'display': format_class_display(predicted_class),
            'confidence': round(confidence, 2),
            'color': get_color_from_class(predicted_class),
            'top3': top3
        })

    except Exception as e:
        return jsonify({'error': f'Kesalahan saat prediksi: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': MODEL is not None,
        'num_classes': len(CLASS_NAMES)
    })


@app.route('/classes')
def get_classes():
    return jsonify({'classes': CLASS_NAMES, 'total': len(CLASS_NAMES)})


if __name__ == '__main__':
    print("\n🃏 UNO Card Classifier — Flask App")
    print("   Buka browser: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
