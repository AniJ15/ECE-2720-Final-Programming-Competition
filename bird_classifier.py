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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
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
    
    # Spectral rolloff (frequency below which 85% of energy is contained)
    cumsum_magnitude = np.cumsum(magnitude)
    rolloff_threshold = 0.85 * cumsum_magnitude[-1] if len(cumsum_magnitude) > 0 else 0
    rolloff_idx = np.where(cumsum_magnitude >= rolloff_threshold)[0]
    if len(rolloff_idx) > 0:
        spectral_rolloff = freqs[rolloff_idx[0]]
    else:
        spectral_rolloff = freqs[-1] if len(freqs) > 0 else 0
    features.append(spectral_rolloff)
    
    # Dominant frequency
    dominant_freq = freqs[np.argmax(magnitude)] if len(magnitude) > 0 else 0
    features.append(dominant_freq)
    
    # Top 3 peak frequencies (important for bird calls)
    if len(magnitude) > 3:
        peak_indices = np.argsort(magnitude)[-3:][::-1]
        top_freqs = freqs[peak_indices]
        features.extend(top_freqs.tolist())
    else:
        features.extend([0, 0, 0])
    
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
    
    # Additional temporal features
    # Variance of amplitude
    features.append(np.var(y_work))
    
    # Mean absolute deviation
    features.append(np.mean(np.abs(y_work - np.mean(y_work))))
    
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
    
    # Spectral contrast (difference between peaks and valleys)
    if len(magnitude) > 10:
        # Simple spectral contrast: difference between mean of top 10% and bottom 10%
        sorted_mag = np.sort(magnitude)
        top_10 = np.mean(sorted_mag[-len(sorted_mag)//10:])
        bottom_10 = np.mean(sorted_mag[:len(sorted_mag)//10])
        spectral_contrast = top_10 - bottom_10 if bottom_10 > 0 else 0
        features.append(spectral_contrast)
    else:
        features.append(0)
    
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


def train_model(X_train, y_train, X_val, y_val, model_type='svm', use_feature_selection=False):
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
        Type of model ('svm', 'knn', 'rf', 'gbm', 'lr', 'ensemble')
    use_feature_selection : bool
        Whether to use feature selection (keeps top 80% of features)
    
    Returns:
    --------
    model : sklearn classifier
        Trained model
    scaler : sklearn scaler
        Fitted scaler
    feature_selector : sklearn feature selector or None
        Feature selector if used
    """
    print(f"\nTraining {model_type.upper()} model...")
    
    # Scale features
    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Optional feature selection (helps prevent overfitting by reducing features)
    feature_selector = None
    if use_feature_selection and X_train_scaled.shape[1] > 20:
        # Use more aggressive feature selection to prevent overfitting
        k_best = max(int(0.7 * X_train_scaled.shape[1]), 20)  # Keep top 70% (more aggressive)
        feature_selector = SelectKBest(f_classif, k=k_best)
        X_train_scaled = feature_selector.fit_transform(X_train_scaled, y_train)
        X_val_scaled = feature_selector.transform(X_val_scaled)
        print(f"  Selected {X_train_scaled.shape[1]} features out of {X_train.shape[1]} (reducing overfitting)")
    
    if model_type == 'svm':
        # Comprehensive SVM tuning with multiple parameters
        print("  Comprehensive SVM tuning (C, gamma, kernel)...")
        best_score = 0
        best_model = None
        best_params = None
        
        # Try different kernels and parameters
        param_grids = [
            {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]},
            {'kernel': ['poly'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'degree': [2, 3]},
            {'kernel': ['sigmoid'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.001, 0.01]}
        ]
        
        for param_grid in param_grids:
            for params in model_selection.ParameterGrid(param_grid):
                try:
                    model = svm.SVC(**params, random_state=42, verbose=False, probability=True)
                    model.fit(X_train_scaled, y_train)
                    val_score = model.score(X_val_scaled, y_val)
                    train_score = model.score(X_train_scaled, y_train)
                    
                    # Penalize large train-val gap
                    gap = train_score - val_score
                    combined_score = val_score - 0.2 * gap
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_model = model
                        best_params = params
                except:
                    continue
        
        train_acc = best_model.score(X_train_scaled, y_train)
        val_acc = best_model.score(X_val_scaled, y_val)
        print(f"  Best params: {best_params} with validation accuracy = {val_acc:.4f}")
        print(f"  Train accuracy: {train_acc:.4f}, Gap: {train_acc - val_acc:.4f}")
        model = best_model
        
    elif model_type == 'knn':
        # Comprehensive KNN tuning
        print("  Comprehensive KNN tuning (k, weights, metric, algorithm)...")
        best_score = 0
        best_model = None
        best_params = None
        
        # Comprehensive parameter search
        k_values = [5, 7, 9, 11, 13, 15, 17, 20, 25, 30]
        weight_options = ['distance', 'uniform']
        metrics = ['euclidean', 'manhattan', 'minkowski']
        algorithms = ['auto', 'ball_tree', 'kd_tree']
        
        for weights in weight_options:
            for metric in metrics:
                for algorithm in algorithms:
                    for k in k_values:
                        try:
                            model = neighbors.KNeighborsClassifier(
                                n_neighbors=k,
                                weights=weights,
                                metric=metric,
                                algorithm=algorithm,
                                leaf_size=30
                            )
                            # Use cross-validation for robust evaluation
                            cv_scores = model_selection.cross_val_score(
                                model, X_train_scaled, y_train,
                                cv=3, scoring='accuracy', n_jobs=-1
                            )
                            cv_mean = cv_scores.mean()
                            
                            model.fit(X_train_scaled, y_train)
                            val_score = model.score(X_val_scaled, y_val)
                            train_score = model.score(X_train_scaled, y_train)
                            
                            gap = train_score - val_score
                            # Combine CV score and validation score, penalize gap
                            combined_score = 0.4 * cv_mean + 0.6 * val_score - 0.3 * gap
                            
                            if combined_score > best_score:
                                best_score = combined_score
                                best_model = model
                                best_params = {'k': k, 'weights': weights, 'metric': metric, 'algorithm': algorithm}
                        except:
                            continue
        
        train_acc = best_model.score(X_train_scaled, y_train)
        val_acc = best_model.score(X_val_scaled, y_val)
        print(f"  Best params: {best_params} with validation accuracy = {val_acc:.4f}")
        print(f"  Train accuracy: {train_acc:.4f}, Gap: {train_acc - val_acc:.4f}")
        model = best_model
        
    elif model_type == 'rf':
        print("  Comprehensive Random Forest tuning...")
        best_score = 0
        best_model = None
        best_params = None
        
        # Comprehensive parameter search
        for n_est in [100, 200, 300, 400]:
            for max_d in [15, 20, 25, 30, None]:  # None = no limit
                for min_split in [2, 5, 10]:
                    for min_leaf in [1, 2, 4]:
                        for max_feat in ['sqrt', 'log2', 0.5]:
                            try:
                                model = RandomForestClassifier(
                                    n_estimators=n_est,
                                    max_depth=max_d,
                                    min_samples_split=min_split,
                                    min_samples_leaf=min_leaf,
                                    max_features=max_feat,
                                    random_state=42,
                                    n_jobs=-1,
                                    verbose=0
                                )
                                model.fit(X_train_scaled, y_train)
                                val_score = model.score(X_val_scaled, y_val)
                                train_score = model.score(X_train_scaled, y_train)
                                
                                gap = train_score - val_score
                                combined_score = val_score - 0.2 * gap
                                
                                if combined_score > best_score:
                                    best_score = combined_score
                                    best_model = model
                                    best_params = {
                                        'n_est': n_est, 'max_d': max_d,
                                        'min_split': min_split, 'min_leaf': min_leaf,
                                        'max_feat': max_feat
                                    }
                            except:
                                continue
        
        train_acc = best_model.score(X_train_scaled, y_train)
        val_acc = best_model.score(X_val_scaled, y_val)
        print(f"  Best params: {best_params} with validation accuracy = {val_acc:.4f}")
        print(f"  Train accuracy: {train_acc:.4f}, Gap: {train_acc - val_acc:.4f}")
        model = best_model
        
    elif model_type == 'gbm':
        print("  Comprehensive Gradient Boosting tuning...")
        best_score = 0
        best_model = None
        best_params = None
        
        # Comprehensive parameter search
        for n_est in [100, 200, 300]:
            for max_d in [3, 4, 5, 6]:
                for lr in [0.01, 0.05, 0.1]:
                    for subsample in [0.8, 0.9, 1.0]:
                        for min_split in [2, 5, 10]:
                            try:
                                model = GradientBoostingClassifier(
                                    n_estimators=n_est,
                                    max_depth=max_d,
                                    learning_rate=lr,
                                    min_samples_split=min_split,
                                    min_samples_leaf=2,
                                    subsample=subsample,
                                    max_features='sqrt',
                                    validation_fraction=0.1,
                                    n_iter_no_change=10,
                                    random_state=42,
                                    verbose=0
                                )
                                model.fit(X_train_scaled, y_train)
                                val_score = model.score(X_val_scaled, y_val)
                                train_score = model.score(X_train_scaled, y_train)
                                
                                gap = train_score - val_score
                                combined_score = val_score - 0.2 * gap
                                
                                if combined_score > best_score:
                                    best_score = combined_score
                                    best_model = model
                                    best_params = {
                                        'n_est': n_est, 'max_d': max_d,
                                        'lr': lr, 'subsample': subsample,
                                        'min_split': min_split
                                    }
                            except:
                                continue
        
        train_acc = best_model.score(X_train_scaled, y_train)
        val_acc = best_model.score(X_val_scaled, y_val)
        print(f"  Best params: {best_params} with validation accuracy = {val_acc:.4f}")
        print(f"  Train accuracy: {train_acc:.4f}, Gap: {train_acc - val_acc:.4f}")
        model = best_model
        
    elif model_type == 'lr':
        print("  Comprehensive Logistic Regression tuning...")
        best_score = 0
        best_model = None
        best_params = None
        
        # Try different regularization types and strengths
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            for penalty in ['l1', 'l2']:
                for solver in ['liblinear', 'lbfgs', 'saga']:
                    # Check solver compatibility
                    if penalty == 'l1' and solver not in ['liblinear', 'saga']:
                        continue
                    if penalty == 'l2' and solver == 'liblinear':
                        continue
                    
                    try:
                        model = LogisticRegression(
                            C=C,
                            penalty=penalty,
                            solver=solver,
                            max_iter=2000,
                            random_state=42,
                            n_jobs=-1
                        )
                        model.fit(X_train_scaled, y_train)
                        val_score = model.score(X_val_scaled, y_val)
                        train_score = model.score(X_train_scaled, y_train)
                        
                        gap = train_score - val_score
                        combined_score = val_score - 0.2 * gap
                        
                        if combined_score > best_score:
                            best_score = combined_score
                            best_model = model
                            best_params = {'C': C, 'penalty': penalty, 'solver': solver}
                    except:
                        continue
        
        train_acc = best_model.score(X_train_scaled, y_train)
        val_acc = best_model.score(X_val_scaled, y_val)
        print(f"  Best params: {best_params} with validation accuracy = {val_acc:.4f}")
        print(f"  Train accuracy: {train_acc:.4f}, Gap: {train_acc - val_acc:.4f}")
        model = best_model
        
    elif model_type == 'ensemble':
        # Weighted ensemble based on individual model performance
        print("  Training weighted ensemble - optimizing individual models first...")
        
        # Train each model individually to get their performance
        individual_models = {}
        individual_scores = {}
        
        print("    Training individual models for ensemble...")
        
        # Train SVM
        print("    - Training SVM...")
        svm_best = None
        svm_best_score = 0
        for C in [1, 10, 100]:
            for gamma in ['scale', 'auto', 0.01]:
                try:
                    m = svm.SVC(kernel='rbf', C=C, gamma=gamma, probability=True, 
                               random_state=42, verbose=False)
                    m.fit(X_train_scaled, y_train)
                    score = m.score(X_val_scaled, y_val)
                    if score > svm_best_score:
                        svm_best_score = score
                        svm_best = m
                except:
                    continue
        individual_models['svm'] = svm_best
        individual_scores['svm'] = svm_best_score
        print(f"      SVM validation accuracy: {svm_best_score:.4f}")
        
        # Train KNN
        print("    - Training KNN...")
        knn_best = None
        knn_best_score = 0
        for k in [11, 15, 20, 25]:
            for weights in ['distance', 'uniform']:
                try:
                    m = neighbors.KNeighborsClassifier(n_neighbors=k, weights=weights, leaf_size=30)
                    m.fit(X_train_scaled, y_train)
                    score = m.score(X_val_scaled, y_val)
                    if score > knn_best_score:
                        knn_best_score = score
                        knn_best = m
                except:
                    continue
        individual_models['knn'] = knn_best
        individual_scores['knn'] = knn_best_score
        print(f"      KNN validation accuracy: {knn_best_score:.4f}")
        
        # Train RF
        print("    - Training Random Forest...")
        rf_best = None
        rf_best_score = 0
        for n_est in [200, 300]:
            for max_d in [20, 25, 30]:
                try:
                    m = RandomForestClassifier(n_estimators=n_est, max_depth=max_d,
                                              min_samples_split=5, min_samples_leaf=2,
                                              max_features='sqrt', random_state=42,
                                              n_jobs=-1, verbose=0)
                    m.fit(X_train_scaled, y_train)
                    score = m.score(X_val_scaled, y_val)
                    if score > rf_best_score:
                        rf_best_score = score
                        rf_best = m
                except:
                    continue
        individual_models['rf'] = rf_best
        individual_scores['rf'] = rf_best_score
        print(f"      RF validation accuracy: {rf_best_score:.4f}")
        
        # Train GBM
        print("    - Training Gradient Boosting...")
        gbm_best = None
        gbm_best_score = 0
        for n_est in [100, 200]:
            for max_d in [4, 5]:
                for lr in [0.05, 0.1]:
                    try:
                        m = GradientBoostingClassifier(n_estimators=n_est, max_depth=max_d,
                                                      learning_rate=lr, subsample=0.8,
                                                      min_samples_split=5, random_state=42, verbose=0)
                        m.fit(X_train_scaled, y_train)
                        score = m.score(X_val_scaled, y_val)
                        if score > gbm_best_score:
                            gbm_best_score = score
                            gbm_best = m
                    except:
                        continue
        individual_models['gbm'] = gbm_best
        individual_scores['gbm'] = gbm_best_score
        print(f"      GBM validation accuracy: {gbm_best_score:.4f}")
        
        # Train LR
        print("    - Training Logistic Regression...")
        lr_best = None
        lr_best_score = 0
        for C in [0.1, 1, 10]:
            try:
                m = LogisticRegression(C=C, max_iter=2000, random_state=42, n_jobs=-1)
                m.fit(X_train_scaled, y_train)
                score = m.score(X_val_scaled, y_val)
                if score > lr_best_score:
                    lr_best_score = score
                    lr_best = m
            except:
                continue
        individual_models['lr'] = lr_best
        individual_scores['lr'] = lr_best_score
        print(f"      LR validation accuracy: {lr_best_score:.4f}")
        
        # Calculate weights based on validation performance
        # Use exponential weighting to emphasize better models
        min_score = min(individual_scores.values())
        max_score = max(individual_scores.values())
        
        # Normalize scores and apply exponential weighting
        if max_score > min_score:
            normalized_scores = {k: (v - min_score) / (max_score - min_score) for k, v in individual_scores.items()}
            # Exponential weighting: better models get much higher weights
            weights = [np.exp(3 * normalized_scores[k]) for k in ['svm', 'knn', 'rf', 'gbm', 'lr']]
            # Normalize weights to sum to number of models
            weights = np.array(weights) * len(weights) / np.sum(weights)
        else:
            weights = [1, 1, 1, 1, 1]  # Equal weights if all perform similarly
        
        print(f"    Model weights: SVM={weights[0]:.2f}, KNN={weights[1]:.2f}, RF={weights[2]:.2f}, GBM={weights[3]:.2f}, LR={weights[4]:.2f}")
        
        # Create weighted ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('svm', individual_models['svm']),
                ('knn', individual_models['knn']),
                ('rf', individual_models['rf']),
                ('gbm', individual_models['gbm']),
                ('lr', individual_models['lr'])
            ],
            voting='soft',
            weights=weights.tolist()
        )
        ensemble.fit(X_train_scaled, y_train)
        score = ensemble.score(X_val_scaled, y_val)
        print(f"  Weighted ensemble validation accuracy = {score:.4f}")
        model = ensemble
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    train_score = model.score(X_train_scaled, y_train)
    val_score = model.score(X_val_scaled, y_val)
    gap = train_score - val_score
    
    print(f"  Final - Train accuracy: {train_score:.4f}, Val accuracy: {val_score:.4f}, Gap: {gap:.4f}")
    
    # Warn if overfitting detected
    if gap > 0.15:
        print(f"  ⚠️  Warning: Large train-val gap ({gap:.4f}) detected - possible overfitting!")
    elif gap > 0.10:
        print(f"  ⚠️  Note: Moderate train-val gap ({gap:.4f}) - monitoring for overfitting")
    
    return model, scaler, feature_selector


def make_predictions(model, scaler, X_test, output_file='submission.csv', feature_selector=None):
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
    feature_selector : sklearn feature selector or None
        Feature selector if used during training
    """
    print("\nMaking predictions on test set...")
    X_test_scaled = scaler.transform(X_test)
    if feature_selector is not None:
        X_test_scaled = feature_selector.transform(X_test_scaled)
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
    # Start with ensemble as it usually performs best
    model_types = ['ensemble', 'svm', 'knn', 'rf', 'gbm', 'lr']
    
    best_model = None
    best_scaler = None
    best_feature_selector = None
    best_score = 0
    best_type = None
    
    for model_type in model_types:
        try:
            # Try without feature selection first
            model, scaler, feature_selector = train_model(
                X_train, y_train, X_val, y_val, 
                model_type=model_type, 
                use_feature_selection=False
            )
            
            # Score with proper feature selection
            X_val_scaled = scaler.transform(X_val)
            if feature_selector is not None:
                X_val_scaled = feature_selector.transform(X_val_scaled)
            score = model.score(X_val_scaled, y_val)
            
            if score > best_score:
                best_score = score
                best_model = model
                best_scaler = scaler
                best_feature_selector = feature_selector
                best_type = model_type
            
            # Save model
            with open(f'model_{model_type}.pkl', 'wb') as f:
                pickle.dump({
                    'model': model, 
                    'scaler': scaler, 
                    'feature_selector': feature_selector
                }, f)
            print(f"Model saved to model_{model_type}.pkl\n")
            
        except Exception as e:
            print(f"Error training {model_type}: {e}\n")
    
    print(f"\nBest model: {best_type} with validation accuracy: {best_score:.4f}")
    
    # Load test data and make predictions
    X_test = load_test_data()
    make_predictions(best_model, best_scaler, X_test, 'submission.csv', best_feature_selector)
    
    print(f"\nTotal time: {time.time() - start_time:.2f}s")

