# AURA - AI: Speech Accent Recognition & Acoustic Analytics

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-green.svg)](https://flask.palletsprojects.com/)
[![Librosa](https://img.shields.io/badge/Audio-Librosa-orange.svg)](https://librosa.org/)
[![scikit-learn](https://img.shields.io/badge/ML-Scikit--Learn-red.svg)](https://scikit-learn.org/)

**AURA - AI** is an end-to-end Machine Learning and Acoustic Signal Processing application designed to classify speech accents from audio inputs. Utilizing **Librosa** for 94-dimensional acoustic feature extraction and a **Support Vector Machine (SVM)** classifier trained on phonetic speech passages, AURA - AI provides real-time accent prediction, probability distributions, and speech metrics through a warm, human-centric web studio.

---

## Table of Contents
- [Key Features](#key-features)
- [Design System & UI Theme](#design-system--ui-theme)
- [Model Performance & Dataset Note](#model-performance--dataset-note)
- [End-to-End Workflow](#end-to-end-workflow)
- [Project Architecture](#project-architecture)
- [System Requirements & Dependencies](#system-requirements--dependencies)
- [Quick Start & Setup Guide](#quick-start--setup-guide)
- [Detailed Audio Pipeline & Resiliency](#detailed-audio-pipeline--resiliency)
- [Supported Accents](#supported-accents)

---

## Key Features

- **Live Microphone Recording**: Capture voice directly in the browser with real-time Web Audio API waveform canvas rendering.
- **Client-Side PCM WAV Encoding**: Converts WebM/Ogg browser recording chunks into 100% standard 16-bit PCM RIFF `.wav` files before uploading.
- **File Upload Studio**: Drag-and-drop `.wav`, `.mp3`, `.webm`, or `.m4a` speech files for acoustic analysis.
- **Multi-Stage Safe Audio Loading**: Backend supports `librosa`, `soundfile`, `pydub`, and `ffmpeg` fallbacks to handle diverse audio encodings cleanly.
- **94-Dimensional Feature Vector**: Extracts MFCCs, Delta & Delta-Delta coefficients, Zero Crossing Rate, Spectral Centroid, Spectral Contrast, Chroma STFT, and RMS Energy.
- **Probabilistic Accent Metrics**: Displays primary accent prediction with an animated SVG score ring and class probability breakdown.
- **Phonetic Passage Reader**: Built-in standard reading passages (e.g., *"Please call Stella..."*) to optimize acoustic consistency.
- **Session Log History**: Persistent browser local storage logging for historical diagnostic comparison.

---

## Design System & UI Theme

AURA - AI features a bespoke, non-AI-generated **Warm Beige & Rough Moss Green** design aesthetic:
- **Canvas & Background**: Warm organic beige (`#F5F2EB`) with cream accents (`#EFEBE3`).
- **Accent Swatches**: Rough Dark Moss (`#2C4632`), Forest Moss (`#3A5A40`), and Sage Green (`#588157`).
- **Typography**: Paired display fonts (*Outfit*), body sans (*Plus Jakarta Sans*), and editorial italic serif (*Instrument Serif*).

---

## Model Performance & Dataset Note

> **Current Model Performance Notice:**
> The machine learning classifier achieves an overall multi-class classification accuracy of approximately **40%** across 10 accent profiles.
> 
> **Why ~40% Accuracy?**
> - **Dataset Skew & Imbalance**: Audio sample counts vary significantly across regional speaker groups, introducing class bias.
> - **Multi-Class Baseline**: Baseline random chance for a 10-class problem is 10%. An accuracy of 40% represents a 4x increase over random probability.
> - **Phonetic Overlap**: Non-native English accents share overlapping spectral centroids and formant characteristics.
>
> **Future Roadmap:**
> Dataset re-balancing, data augmentation (noise injection & pitch shifting), and deep learning architectures (Mel-Spectrogram CNNs, Wav2Vec 2.0, or Whisper embeddings).

---

## End-to-End Workflow

```
┌──────────────────────────────────────────────┐
│  Dataset: speakers_all.csv + recordings/     │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  1. Feature Extraction                       │
│     (ml/feature_engineering/extract_features)│
│     • Resample to 16 kHz & normalize         │
│     • Compute 94 audio features (MFCC, ZCR)  │
│     • Output: data/processed/features.csv    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  2. Model Training & Evaluation              │
│     (ml/training/train_model.py)             │
│     • 80/20 Train-Test Stratified Split     │
│     • StandardScaler + RBF Kernel SVM        │
│     • Output: models/best_model.joblib       │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  3. Flask Web Service & Real-Time Inference  │
│     (backend/app.py)                         │
│     • Live Web Audio PCM WAV recording       │
│     • Multi-stage audio loader fallback      │
│     • REST Endpoint POST /predict            │
│     • Real-time feature calculation & result │
└──────────────────────────────────────────────┘
```

---

## Project Architecture

```text
ai-accent-detector/
├── backend/
│   └── app.py                     # Flask server & safe multi-stage audio processing
├── data/
│   ├── raw/                       # Raw input data references
│   └── processed/
│       └── features.csv           # Generated 94-feature dataset
├── ml/
│   ├── feature_engineering/
│   │   └── extract_features.py    # Librosa feature extraction script
│   └── training/
│       ├── train_model.py         # Model training script (SVM & Random Forest)
│       └── evaluate_model.py      # Classifier metric evaluation script
├── models/
│   ├── best_model.joblib          # Trained SVM pipeline model
│   └── rf_model.joblib            # Trained Random Forest comparison model
├── recordings/                    # Audio dataset (.mp3 files)
├── reports/                       # Training reports & evaluation graphics
├── static/
│   ├── css/
│   │   └── style.css              # Warm Beige & Rough Moss Green stylesheet
│   └── js/
│       └── main.js                # PCM WAV encoding, visualizer, API handlers
├── templates/
│   └── index.html                 # Frontend Web Application (Aura - AI)
├── reading-passage.txt            # Benchmark speech prompt text
├── requirements.txt               # Python package dependencies
├── setup_and_train.bat            # Automated 1-click Windows setup script
└── speakers_all.csv               # Speaker metadata (country, native language)
```

---

## System Requirements & Dependencies

### Prerequisites
- **Python**: Version `3.8` to `3.12`
- **Pip**: Latest version
- **OS**: Windows, macOS, or Linux

### Python Dependencies (`requirements.txt`)
| Package | Description |
|---|---|
| `numpy` | High-performance numerical and vector operations |
| `pandas` | CSV parsing and data frame management |
| `librosa` | Audio signal processing and acoustic feature extraction |
| `soundfile` | Audio reading backend |
| `pydub` | Fallback audio format decoding |
| `scipy` | Signal processing subroutines |
| `scikit-learn` | Standard Scaling, SVM model training, and evaluation |
| `flask` | Lightweight web backend REST server |
| `flask-cors` | Cross-Origin Resource Sharing handling |
| `joblib` | Model persistence & serialization |

---

## Quick Start & Setup Guide

### Option 1: Automated Setup (Windows)
Run the included batch script in your terminal:

```cmd
.\setup_and_train.bat
```

---

### Option 2: Manual Setup

1. **Clone the repository and enter project folder:**
   ```bash
   git clone https://github.com/your-repo/ai-accent-detector.git
   cd ai-accent-detector
   ```

2. **Create and activate a Virtual Environment:**
   - **Windows:**
     ```cmd
     python -m venv venv
     venv\Scripts\activate
     ```
   - **macOS / Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install pydub
   ```

4. **Run Feature Extraction:**
   ```bash
   python ml/feature_engineering/extract_features.py
   ```

5. **Train the ML Model:**
   ```bash
   python ml/training/train_model.py
   ```

6. **Launch the Web Application:**
   ```bash
   python backend/app.py
   ```

7. **Open in Browser:**
   Navigate to `http://localhost:5000` in your web browser.

---

## Detailed Audio Pipeline & Resiliency

### 1. Client-Side PCM WAV Encoding
Browsers natively record audio in WebM or Ogg format via `MediaRecorder`. AURA - AI decodes recorded audio chunks in `main.js` via the Web Audio API (`AudioContext.decodeAudioData()`) and compiles a 100% compliant **16-bit Mono PCM RIFF `.wav` file** prior to HTTP upload.

### 2. Multi-Stage Backend Safe Loader
In `backend/app.py`, `load_audio_safely()` executes a multi-layered fallback strategy:
1. `librosa.load()` for native WAV/FLAC files.
2. `soundfile.read()` for raw soundfile streams.
3. `pydub.AudioSegment` for compressed WebM, MP3, and M4A uploads.
4. `ffmpeg` CLI invocation if available on host system PATH.

### 3. Feature Extraction Vector (94 Dimensions)
- **MFCCs (13 coefficients)**: Mean & Std (26 features)
- **Delta MFCCs (13 coefficients)**: Mean & Std (26 features)
- **Delta-2 MFCCs (13 coefficients)**: Mean & Std (26 features)
- **Zero Crossing Rate (ZCR)**: Mean & Std (2 features)
- **Spectral Centroid**: Mean & Std (2 features)
- **Spectral Contrast (7 bands)**: Mean & Std (14 features)
- **RMS Energy**: Mean & Std (2 features)

---

## Supported Accents

- American English
- British English
- Canadian English
- Australian English
- Indian English
- Nepali English
- Chinese English
- Arabic English
- Russian English
- Japanese English

## Screenshot
<img width="2477" height="1467" alt="image" src="https://github.com/user-attachments/assets/547de62b-eca2-4127-91de-717c2f845b82" />
