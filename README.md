# AURA.AI: AI-Powered Accent Recognition System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-green.svg)](https://flask.palletsprojects.com/)
[![Librosa](https://img.shields.io/badge/Audio-Librosa-orange.svg)](https://librosa.org/)
[![scikit-learn](https://img.shields.io/badge/ML-Scikit--Learn-red.svg)](https://scikit-learn.org/)

**AURA.AI** is an end-to-end Machine Learning and Audio Processing application designed to classify non-native and native English accents from speech audio. Using **Librosa** for 94-dimensional acoustic feature extraction and **Support Vector Machines (SVM)** trained on standard speech passages, AURA.AI delivers real-time accent predictions through a web interface.

---

## Table of Contents
- [Key Features](#key-features)
- [Model Performance & Accuracy Note](#model-performance--accuracy-note)
- [End-to-End Workflow](#end-to-end-workflow)
- [Project Architecture](#project-architecture)
- [System Requirements & Dependencies](#system-requirements--dependencies)
- [Quick Start & Setup Guide](#quick-start--setup-guide)
- [Detailed Pipeline Explanation](#detailed-pipeline-explanation)
  - [1. Acoustic Feature Extraction](#1-acoustic-feature-extraction)
  - [2. Model Training & Evaluation](#2-model-training--evaluation)
  - [3. Web Application & Inference](#3-web-application--inference)
- [Supported Accents](#supported-accents)

---

## Key Features

- **Live Voice Recording**: Record reading passages directly in your web browser with dynamic Canvas audio waveform visualization.
- **File Upload**: Drag-and-drop `.wav`, `.mp3`, or `.m4a` speech files for instant acoustic analysis.
- **94-Dimensional Acoustic Profiling**: Extracts MFCCs, Delta & Delta-Delta coefficients, Zero Crossing Rate, Spectral Centroid, Spectral Contrast, Chroma STFT, and RMS Energy.
- **Probabilistic Classification**: Displays primary accent detection with a confidence meter and full probability distribution breakdown.
- **Passage Selector**: Built-in standard phonetic reading passages (e.g., *"Please call Stella..."*) to optimize recognition accuracy.
- **In-Browser Session History**: Track recent predictions with local storage history logging.

---

## Model Performance & Accuracy Note

> **Current Model Performance Notice:**
> The current machine learning classifier achieves an overall multi-class classification accuracy of approximately **40%** across 10 accent classes.
> 
> **Why ~40% Accuracy?**
> - **Highly Skewed Dataset**: The primary reason for the limited accuracy is that the underlying dataset is highly skewed and imbalanced across speaker groups. Certain accents have significantly more audio samples than others, causing model bias.
> - **Multi-Class Complexity**: Random guessing across 10 classes yields a baseline accuracy of only 10%. An accuracy of 40% is 4x higher than random chance.
> - **Acoustic Overlap**: Non-native English accents often share subtle acoustic and phonetic similarities.
>
> **Path to Higher Accuracy:**
> With a balanced, clean, and non-skewed dataset, the classification performance can be significantly improved. Future work includes expanding dataset balance across all regional speaker groups and utilizing deep learning architectures (such as CNNs on Mel-Spectrograms or Wav2Vec 2.0 / Whisper embeddings).

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
│     • Live Web Audio API recording           │
│     • REST Endpoint POST /predict            │
│     • Real-time feature calculation & result │
└──────────────────────────────────────────────┘
```

---

## Project Architecture

```text
ai-accent-detector/
├── backend/
│   └── app.py                     # Flask REST server & inference endpoints
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
│   └── rf_model.joblib            # Trained Random Forest model (comparison)
├── recordings/                    # Audio dataset (.mp3 files)
├── reports/                       # Training reports & comparison graphics
├── static/
│   ├── css/
│   │   └── style.css              # Glassmorphic UI styling
│   └── js/
│       └── main.js                # MediaRecorder, visualizer, API logic
├── templates/
│   └── index.html                 # Frontend Web UI
├── reading-passage.txt            # Benchmark speech prompt text
├── requirements.txt               # Python package dependencies
├── setup_and_train.bat            # Automated 1-click Windows setup script
└── speakers_all.csv               # Speaker metadata (country, native language)
```

---

## System Requirements & Dependencies

### Prerequisites
- **Python**: Version `3.8` to `3.11`
- **Pip**: Latest version
- **OS**: Windows, macOS, or Linux

### Python Dependencies (`requirements.txt`)
| Package | Description |
|---|---|
| `numpy` | High-performance array operations |
| `pandas` | CSV parsing and data frame management |
| `librosa` | Audio signal processing and acoustic feature extraction |
| `soundfile` | Sound file reading and writing backend |
| `scipy` | Signal processing subroutines |
| `scikit-learn` | Standard Scaling, SVM model training, and evaluation |
| `flask` | Lightweight web backend REST framework |
| `flask-cors` | Cross-Origin Resource Sharing handling |
| `matplotlib` | Plotting evaluation graphs and comparison metrics |

---

## Quick Start & Setup Guide

### Option 1: Automated 1-Click Setup (Windows)
Run the included batch script in your terminal to set up the environment, extract features, train the model, and launch the web app:

```cmd
.\setup_and_train.bat
```

---

### Option 2: Manual Step-by-Step Setup

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
   ```

4. **Run Feature Extraction:**
   ```bash
   python ml/feature_engineering/extract_features.py
   ```

5. **Train the ML Model:**
   ```bash
   python ml/training/train_model.py
   ```

6. **Evaluate the Model (Optional):**
   ```bash
   python ml/training/evaluate_model.py
   ```

7. **Launch the Web Application:**
   ```bash
   python backend/app.py
   ```

8. **Open in Browser:**
   Navigate to `http://localhost:5000` in your web browser.

---

## Detailed Pipeline Explanation

### 1. Acoustic Feature Extraction
Audio signals are normalized and resampled to **16 kHz**. From each audio file, 94 features are calculated:
- **MFCC (0-12)**: Mean & standard deviation of 13 Mel-Frequency Cepstral Coefficients (26 features).
- **Delta MFCC (0-12)**: 1st order derivative (velocity) (26 features).
- **Delta-2 MFCC (0-12)**: 2nd order derivative (acceleration) (26 features).
- **Zero Crossing Rate (ZCR)**: Mean & std (2 features).
- **Spectral Centroid**: Mean & std of center of mass of the spectrum (2 features).
- **Spectral Contrast (7 bands)**: Mean & std across 6 sub-bands + total (14 features).
- **Chroma STFT (12 pitch classes)**: Mean & std (2 features summarized/processed).
- **RMS Energy**: Mean & std of signal energy (2 features).

### 2. Model Training & Evaluation
- The extracted feature set is scaled using `StandardScaler`.
- Training uses an **RBF-kernel Support Vector Classifier (SVC)** with `probability=True` for soft probability outputs.
- A **Random Forest Classifier** is also trained for model comparison and reporting.

### 3. Web Application & Inference
- The browser captures speech via the `MediaRecorder` API as an audio blob.
- The blob is sent via HTTP `POST` to `/predict`.
- Flask processes the temporary file with Librosa, extracts the 94-feature vector, and passes it to `best_model.joblib`.
- Results are returned as JSON containing the **top accent**, **confidence score**, and **probability distribution matrix**.

---

## Supported Accents

The current model classifies speech into the following accent profiles based on speaker origin and native language:

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
