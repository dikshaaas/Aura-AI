import os
import numpy as np
import pandas as pd
import librosa
import warnings

# Suppress librosa/audioread warnings
warnings.filterwarnings('ignore')

def get_accent(native_lang, country):
    native_lang = str(native_lang).lower().strip()
    country = str(country).lower().strip()
    
    if native_lang == 'english':
        if country == 'usa':
            return 'American English'
        elif country == 'uk':
            return 'British English'
        elif country == 'canada':
            return 'Canadian English'
        elif country == 'australia':
            return 'Australian English'
    
    if native_lang == 'arabic':
        return 'Arabic English'
    
    if native_lang in ['mandarin', 'cantonese'] or country in ['china', 'taiwan']:
        return 'Chinese English'
        
    if country == 'india':
        return 'Indian English'
        
    if native_lang == 'japanese':
        return 'Japanese English'
        
    if native_lang in ['nepali', 'newari'] or country == 'nepal':
        return 'Nepali English'
        
    if native_lang == 'russian':
        return 'Russian English'
        
    return None

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
    features = {}
    
    # MFCCs
    for i in range(13):
        features[f'mfcc_mean_{i}'] = np.mean(mfcc[i])
        features[f'mfcc_std_{i}'] = np.std(mfcc[i])
        
    # Delta
    for i in range(13):
        features[f'delta_mean_{i}'] = np.mean(delta[i])
        features[f'delta_std_{i}'] = np.std(delta[i])
        
    # Delta2
    for i in range(13):
        features[f'delta2_mean_{i}'] = np.mean(delta2[i])
        features[f'delta2_std_{i}'] = np.std(delta2[i])
        
    # ZCR
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    # Centroid
    features['centroid_mean'] = np.mean(centroid)
    features['centroid_std'] = np.std(centroid)
    
    # Chroma
    for i in range(12):
        features[f'chroma_mean_{i}'] = np.mean(chroma[i])
        features[f'chroma_std_{i}'] = np.std(chroma[i])
        
    # Contrast
    for i in range(7):
        features[f'contrast_mean_{i}'] = np.mean(contrast[i])
        features[f'contrast_std_{i}'] = np.std(contrast[i])
        
    # RMS
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    
    return features

def main():
    print("Initializing feature extraction...")
    
    speakers_csv_path = "speakers_all.csv"
    recordings_dir = os.path.join("recordings", "recordings")
    output_csv_path = os.path.join("data", "processed", "features.csv")
    
    # Create target directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    if not os.path.exists(speakers_csv_path):
        print(f"Error: {speakers_csv_path} not found!")
        return
        
    df_speakers = pd.read_csv(speakers_csv_path)
    
    rows = []
    
    # Find all files in recordings directory
    if not os.path.exists(recordings_dir):
        print(f"Error: Recordings directory {recordings_dir} not found!")
        return
        
    audio_files = [f for f in os.listdir(recordings_dir) if f.endswith('.mp3') or f.endswith('.wav')]
    print(f"Found {len(audio_files)} audio files in recordings directory.")
    
    count = 0
    for idx, row in df_speakers.iterrows():
        filename = str(row['filename'])
        # Add .mp3 extension
        filename_ext = filename + '.mp3'
        
        # Check if file exists in our recordings directory
        file_path = os.path.join(recordings_dir, filename_ext)
        if not os.path.exists(file_path):
            continue
            
        # Determine target accent
        accent = get_accent(row['native_language'], row['country'])
        if accent is None:
            continue
            
        print(f"[{count+1}] Extracting features for {filename_ext} ({accent})...")
        
        try:
            y, sr = librosa.load(file_path, sr=16000)
            feat = extract_audio_features(y, sr)
            feat['accent'] = accent
            feat['file'] = filename_ext
            rows.append(feat)
            count += 1
        except Exception as e:
            print(f"Error processing {filename_ext}: {e}")
            
    if len(rows) == 0:
        print("No features extracted.")
        return
        
    df_features = pd.DataFrame(rows)
    
    # Reorder columns to match the expected format
    columns_order = []
    # MFCCs
    for i in range(13):
        columns_order.extend([f'mfcc_mean_{i}', f'mfcc_std_{i}'])
    # Delta
    for i in range(13):
        columns_order.extend([f'delta_mean_{i}', f'delta_std_{i}'])
    # Delta2
    for i in range(13):
        columns_order.extend([f'delta2_mean_{i}', f'delta2_std_{i}'])
    # ZCR, Centroid
    columns_order.extend(['zcr_mean', 'zcr_std', 'centroid_mean', 'centroid_std'])
    # Chroma
    for i in range(12):
        columns_order.extend([f'chroma_mean_{i}', f'chroma_std_{i}'])
    # Contrast
    for i in range(7):
        columns_order.extend([f'contrast_mean_{i}', f'contrast_std_{i}'])
    # RMS
    columns_order.extend(['rms_mean', 'rms_std', 'accent', 'file'])
    
    # Reindex columns
    df_features = df_features.reindex(columns=columns_order)
    
    df_features.to_csv(output_csv_path, index=False)
    print(f"Successfully saved {len(df_features)} feature rows to {output_csv_path}")

if __name__ == "__main__":
    main()
