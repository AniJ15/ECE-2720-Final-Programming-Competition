import os
import csv
import time
import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.stats import skew, kurtosis
from tqdm import tqdm
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import pickle


def extract_spectral_features(y, sr=44100):
    """
    Extract optimized spectral features from audio signal.
    Faster version with reduced computations.
    
    Parameters:
    -----------
    y : ndarray
        Audio signal
    sr : int
        Sample rate (default: 44100)
    
    Returns:
    --------
    features : ndarray
        Extracted features
    """
    # Downsample if signal is too long (faster FFT) - work on copy
    y_work = y.copy()
    sr_work = sr
    if len(y_work) > 22050:  # If more than 0.5s at 44.1kHz
        # Downsample by taking every nth sample
        step = len(y_work) // 22050
        y_work = y_work[::step]
        sr_work = sr_work // step
    
    # Single FFT computation
    fft_data = np.fft.fft(y_work)
    magnitude = np.abs(fft_data[:len(fft_data)//2])
    
    # Basic statistics (faster - fewer percentiles)
    features = [
        np.mean(magnitude),
        np.median(magnitude),
        np.std(magnitude),
        np.max(magnitude),
        np.min(magnitude),
    ]
    
    # Frequency domain features
    freqs = np.fft.fftfreq(len(y_work), 1/sr_work)[:len(magnitude)]
    
    # Spectral centroid
    total_mag = np.sum(magnitude)
    if total_mag > 0:
        spectral_centroid = np.sum(freqs * magnitude) / total_mag
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / total_mag)
    else:
        spectral_centroid = 0
        spectral_bandwidth = 0
    
    features.extend([spectral_centroid, spectral_bandwidth])
    
    # Dominant frequency
    dominant_freq = freqs[np.argmax(magnitude)]
    features.append(dominant_freq)
    
    # Energy in frequency bands (vectorized, faster)
    freq_mask_1 = (freqs >= 0) & (freqs < 1000)
    freq_mask_2 = (freqs >= 1000) & (freqs < 3000)
    freq_mask_3 = (freqs >= 3000) & (freqs < 6000)
    freq_mask_4 = (freqs >= 6000) & (freqs < 10000)
    freq_mask_5 = freqs >= 10000
    
    if total_mag > 0:
        features.extend([
            np.sum(magnitude[freq_mask_1]) / total_mag,
            np.sum(magnitude[freq_mask_2]) / total_mag,
            np.sum(magnitude[freq_mask_3]) / total_mag,
            np.sum(magnitude[freq_mask_4]) / total_mag,
            np.sum(magnitude[freq_mask_5]) / total_mag,
        ])
    else:
        features.extend([0, 0, 0, 0, 0])
    
    return np.array(features)


def extract_temporal_features(y):
    """
    Extract optimized temporal features from audio signal.
    Faster version without expensive autocorrelation.
    
    Parameters:
    -----------
    y : ndarray
        Audio signal
    
    Returns:
    --------
    features : ndarray
        Extracted features
    """
    # Downsample for faster computation if needed (work on copy)
    y_work = y.copy()
    if len(y_work) > 22050:
        step = len(y_work) // 22050
        y_work = y_work[::step]
    
    # Basic statistics (vectorized)
    features = [
        np.mean(y_work),
        np.std(y_work),
        np.max(y_work),
        np.min(y_work),
        np.ptp(y_work),  # peak-to-peak
    ]
    
    # Zero crossing rate (optimized)
    sign_changes = np.diff(np.signbit(y_work))
    zcr = np.sum(sign_changes != 0) / len(y_work)
    features.append(zcr)
    
    # Root mean square energy
    rms = np.sqrt(np.mean(y_work**2))
    features.append(rms)
    
    # Simple energy measure
    energy = np.sum(y_work**2)
    features.append(energy)
    
    return np.array(features)


def extract_simple_spectral_features(y, sr=44100):
    """
    Extract simple additional spectral features (replaces expensive MFCC).
    Much faster alternative that still captures frequency characteristics.
    
    Parameters:
    -----------
    y : ndarray
        Audio signal
    sr : int
        Sample rate
    
    Returns:
    --------
    features : ndarray
        Extracted features
    """
    # Downsample if needed (work on copy)
    y_work = y.copy()
    sr_work = sr
    if len(y_work) > 22050:
        step = len(y_work) // 22050
        y_work = y_work[::step]
        sr_work = sr_work // step
    
    # Single FFT
    fft_data = np.fft.fft(y_work)
    magnitude = np.abs(fft_data[:len(fft_data)//2])
    freqs = np.fft.fftfreq(len(y_work), 1/sr_work)[:len(magnitude)]
    
    # Simple frequency band energies (coarser, faster)
    # Divide into 8 bands instead of complex mel filter bank
    n_bands = 8
    max_freq = sr_work / 2
    band_edges = np.linspace(0, max_freq, n_bands + 1)
    
    features = []
    total_energy = np.sum(magnitude)
    
    if total_energy > 0:
        for i in range(n_bands):
            mask = (freqs >= band_edges[i]) & (freqs < band_edges[i+1])
            band_energy = np.sum(magnitude[mask]) / total_energy
            features.append(band_energy)
    else:
        features = [0] * n_bands
    
    # Add spectral flatness (measure of noisiness)
    if np.mean(magnitude) > 0:
        spectral_flatness = np.exp(np.mean(np.log(magnitude + 1e-10))) / np.mean(magnitude)
    else:
        spectral_flatness = 0
    features.append(spectral_flatness)
    
    return np.array(features)


def extract_all_features(y, sr=44100):
    """
    Extract all features from audio signal (optimized version).
    
    Parameters:
    -----------
    y : ndarray
        Audio signal
    sr : int
        Sample rate
    
    Returns:
    --------
    features : ndarray
        Combined feature vector
    """
    # Normalize audio to prevent issues with different scales
    if len(y) > 0:
        y = y.astype(np.float64)
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
    
    spectral = extract_spectral_features(y, sr)
    temporal = extract_temporal_features(y)
    simple_spectral = extract_simple_spectral_features(y, sr)
    
    return np.concatenate([spectral, temporal, simple_spectral])


def load_training_data(data_dir='./kaggle_upload/train/train', labels_file='./kaggle_upload/train.csv'):
    """
    Load training data and labels.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing training audio files
    labels_file : str
        Path to CSV file with training labels
    
    Returns:
    --------
    X : ndarray
        Feature matrix
    y : ndarray
        Label vector
    """
    print("Loading training audio files...")
    train_frames = []
    sample_rates = []
    missing_files = []
    
    for i in tqdm(range(20000), desc='Loading audio', leave=False):
        filepath = os.path.join(data_dir, f'train_{i}.wav')
        if os.path.exists(filepath):
            sr, frames = wavfile.read(filepath)
            train_frames.append(frames)
            sample_rates.append(sr)
        else:
            missing_files.append(i)
            train_frames.append(np.zeros(44100 * 3))  # 3 seconds at 44.1kHz
            sample_rates.append(44100)
    
    if missing_files:
        print(f"Warning: {len(missing_files)} training files not found (e.g., train_{missing_files[0]}.wav)")
    
    print("Loading training labels...")
    with open(labels_file, 'r') as f:
        reader = csv.reader(f)
        ls = list(reader)
        train_y = np.array(ls)
        train_y = train_y[1:, 1].astype(int)
    
    print("Extracting features...")
    train_features = []
    for i, frames in enumerate(tqdm(train_frames, desc='Extracting features', leave=False)):
        features = extract_all_features(frames, sr=sample_rates[i])
        train_features.append(features)
    
    X = np.array(train_features)
    y = train_y
    
    return X, y


def load_test_data(data_dir='./kaggle_upload/test/test'):
    """
    Load test data.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing test audio files
    
    Returns:
    --------
    X : ndarray
        Feature matrix
    """
    print("Loading test audio files...")
    test_frames = []
    sample_rates = []
    missing_files = []
    
    for i in tqdm(range(4000), desc='Loading audio', leave=False):
        filepath = os.path.join(data_dir, f'test_{i}.wav')
        if os.path.exists(filepath):
            sr, frames = wavfile.read(filepath)
            test_frames.append(frames)
            sample_rates.append(sr)
        else:
            missing_files.append(i)
            test_frames.append(np.zeros(44100 * 3))
            sample_rates.append(44100)
    
    if missing_files:
        print(f"Warning: {len(missing_files)} test files not found (e.g., test_{missing_files[0]}.wav)")
    
    print("Extracting features...")
    test_features = []
    for i, frames in enumerate(tqdm(test_frames, desc='Extracting features', leave=False)):
        features = extract_all_features(frames, sr=sample_rates[i])
        test_features.append(features)
    
    X = np.array(test_features)
    return X


def train_model(X_train, y_train, X_val, y_val, model_type='svm'):
    """
    Train a classifier model.
    
    Parameters:
    -----------
    X_train : ndarray
        Training features
    y_train : ndarray
        Training labels
    X_val : ndarray
        Validation features
    y_val : ndarray
        Validation labels
    model_type : str
        Type of model ('svm', 'knn', 'rf', 'ensemble')
    
    Returns:
    --------
    model : sklearn classifier
        Trained model
    scaler : sklearn scaler
        Fitted scaler
    """
    print(f"\nTraining {model_type.upper()} model...")
    
    # Scale features
    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    if model_type == 'svm':
        # Tune C parameter
        best_score = 0
        best_model = None
        best_C = None
        
        print("  Tuning C parameter...")
        for C in [0.1, 1, 10, 100, 1000]:
            model = svm.SVC(kernel='rbf', C=C, gamma='scale', random_state=42, verbose=False)
            model.fit(X_train_scaled, y_train)
            score = model.score(X_val_scaled, y_val)
            if score > best_score:
                best_score = score
                best_model = model
                best_C = C
        
        print(f"  Best C={best_C} with validation accuracy = {best_score:.4f}")
        model = best_model
        
    elif model_type == 'knn':
        # Tune k parameter
        best_score = 0
        best_model = None
        best_k = None
        
        print("  Tuning k parameter...")
        for k in [3, 5, 7, 9, 11, 15, 20, 25]:
            model = neighbors.KNeighborsClassifier(n_neighbors=k, weights='distance')
            model.fit(X_train_scaled, y_train)
            score = model.score(X_val_scaled, y_val)
            if score > best_score:
                best_score = score
                best_model = model
                best_k = k
        
        print(f"  Best k={best_k} with validation accuracy = {best_score:.4f}")
        model = best_model
        
    elif model_type == 'rf':
        print("  Training Random Forest...")
        model = RandomForestClassifier(n_estimators=200, max_depth=30, 
                                      random_state=42, n_jobs=-1, verbose=0)
        model.fit(X_train_scaled, y_train)
        score = model.score(X_val_scaled, y_val)
        print(f"  Validation accuracy = {score:.4f}")
        
    elif model_type == 'ensemble':
        # Ensemble of multiple models
        print("  Training ensemble (SVM + KNN + RF)...")
        svm_model = svm.SVC(kernel='rbf', C=100, gamma='scale', 
                           probability=True, random_state=42, verbose=False)
        knn_model = neighbors.KNeighborsClassifier(n_neighbors=11, weights='distance')
        rf_model = RandomForestClassifier(n_estimators=200, max_depth=30, 
                                         random_state=42, n_jobs=-1, verbose=0)
        
        ensemble = VotingClassifier(
            estimators=[('svm', svm_model), ('knn', knn_model), ('rf', rf_model)],
            voting='soft'
        )
        ensemble.fit(X_train_scaled, y_train)
        score = ensemble.score(X_val_scaled, y_val)
        print(f"  Validation accuracy = {score:.4f}")
        model = ensemble
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    train_score = model.score(X_train_scaled, y_train)
    val_score = model.score(X_val_scaled, y_val)
    print(f"  Final - Train accuracy: {train_score:.4f}, Val accuracy: {val_score:.4f}")
    
    return model, scaler


def make_predictions(model, scaler, X_test, output_file='submission.csv'):
    """
    Make predictions on test set and save to CSV.
    
    Parameters:
    -----------
    model : sklearn classifier
        Trained model
    scaler : sklearn scaler
        Fitted scaler
    X_test : ndarray
        Test features
    output_file : str
        Output CSV file path
    """
    print("\nMaking predictions on test set...")
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    
    ids = np.arange(0, predictions.shape[0])
    output = np.transpose(np.vstack((ids, predictions))).astype(int)
    header = np.array([['# ID', 'Label']])
    output = np.vstack((header, output))
    
    np.savetxt(output_file, output, delimiter=',', fmt='%s')
    print(f"Predictions saved to {output_file}")


if __name__ == '__main__':
    # Load training data
    start_time = time.time()
    X, y = load_training_data()
    print(f"\nData loading time: {time.time() - start_time:.2f}s")
    print(f"Feature shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = model_selection.train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\nTrain set: {X_train.shape}, Validation set: {X_val.shape}")
    
    # Train model (try different models)
    model_types = ['svm', 'knn', 'rf', 'ensemble']
    
    best_model = None
    best_scaler = None
    best_score = 0
    best_type = None
    
    for model_type in model_types:
        try:
            model, scaler = train_model(X_train, y_train, X_val, y_val, model_type=model_type)
            score = model.score(scaler.transform(X_val), y_val)
            
            if score > best_score:
                best_score = score
                best_model = model
                best_scaler = scaler
                best_type = model_type
            
            # Save model
            with open(f'model_{model_type}.pkl', 'wb') as f:
                pickle.dump({'model': model, 'scaler': scaler}, f)
            print(f"Model saved to model_{model_type}.pkl\n")
            
        except Exception as e:
            print(f"Error training {model_type}: {e}\n")
    
    print(f"\nBest model: {best_type} with validation accuracy: {best_score:.4f}")
    
    # Load test data and make predictions
    X_test = load_test_data()
    make_predictions(best_model, best_scaler, X_test, 'submission.csv')
    
    print(f"\nTotal time: {time.time() - start_time:.2f}s")

