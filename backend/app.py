import os
import tempfile
import warnings
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import librosa
import joblib
import numpy as np

# Suppress librosa/audioread warnings
warnings.filterwarnings('ignore')

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), '..', 'static')
)
CORS(app)

# Load the trained model pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.joblib')
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Model path {MODEL_PATH} not found. Start server anyway...")

def extract_audio_features(y, sr):
    # Normalize audio signal
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
        
    # Extract features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    rms = librosa.feature.rms(y=y)
    
    # Compile feature dictionary
    features = []
    
    # MFCCs
    for i in range(13):
        features.extend([np.mean(mfcc[i]), np.std(mfcc[i])])
        
    # Delta
    for i in range(13):
        features.extend([np.mean(delta[i]), np.std(delta[i])])
        
    # Delta2
    for i in range(13):
        features.extend([np.mean(delta2[i]), np.std(delta2[i])])
        
    # ZCR, Centroid
    features.extend([np.mean(zcr), np.std(zcr), np.mean(centroid), np.std(centroid)])
    
    # Chroma
    for i in range(12):
        features.extend([np.mean(chroma[i]), np.std(chroma[i])])
        
    # Contrast
    for i in range(7):
        features.extend([np.mean(contrast[i]), np.std(contrast[i])])
        
    # RMS
    features.extend([np.mean(rms), np.std(rms)])
    
    return np.array(features).reshape(1, -1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/passages')
def get_passages():
    # Passages expected by frontend (with id, theme, text)
    return jsonify([
        {
            "id": "stella",
            "theme": "Please Call Stella (Standard)",
            "text": "Please call Stella. Ask her to bring these things with her from the store: Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob. We also need a small plastic snake and a big toy frog for the kids. She can scoop these things into three red bags, and we will go meet her Wednesday at the train station."
        }
    ])

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        load_model()
        if model is None:
            return jsonify({
                "status": "error",
                "message": "Model not trained yet or unavailable. Please run the ML training pipeline first."
            }), 500
            
    if 'file' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No file uploaded."
        }), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            "status": "error",
            "message": "No selected file."
        }), 400
        
    # Save the file temporarily in the workspace directory
    temp_dir = os.path.join(os.path.dirname(__file__), 'temp_uploads')
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, next(tempfile._get_candidate_names()) + '.wav')
    
    try:
        file.save(temp_file_path)
        
        # Load and process audio
        y, sr = librosa.load(temp_file_path, sr=16000)
        
        # Extract features
        features_vec = extract_audio_features(y, sr)
        
        # Run prediction
        probs = model.predict_proba(features_vec)[0]
        classes = model.classes_
        
        # Sort classes and probabilities by probability descending
        distribution = {}
        sorted_indices = np.argsort(probs)[::-1]
        for idx in sorted_indices:
            distribution[classes[idx]] = round(float(probs[idx]) * 100, 1)
            
        best_class = classes[sorted_indices[0]]
        best_prob = round(float(probs[sorted_indices[0]]) * 100)
        
        return jsonify({
            "status": "success",
            "accent": best_class,
            "confidence": best_prob,
            "distribution": distribution
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            "status": "error",
            "message": f"Acoustic analysis failed: {str(e)}"
        }), 500
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                print(f"Error removing temp file {temp_file_path}: {e}")

if __name__ == "__main__":
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
