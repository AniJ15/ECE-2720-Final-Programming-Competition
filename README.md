# Bird Call Classification Competition

This project implements a machine learning solution for classifying bird calls from 3-second audio samples.

## Overview

The competition involves classifying bird calls from audio segments. The solution uses traditional machine learning approaches (no neural networks) with comprehensive feature extraction from audio signals.

## Features

- **Comprehensive Feature Extraction**:
  - Spectral features (FFT-based statistics, spectral centroid, rolloff, bandwidth)
  - Temporal features (zero crossing rate, RMS energy, autocorrelation)
  - MFCC-like features (approximated without librosa)
  - Frequency band energy ratios

- **Multiple Classifier Options**:
  - Support Vector Machine (SVM) with RBF kernel
  - K-Nearest Neighbors (KNN)
  - Random Forest
  - Ensemble of all three models

- **Automatic Hyperparameter Tuning**: The script automatically tunes hyperparameters for each model type

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the competition data from Kaggle:
```bash
kaggle competitions download -c fa-25-ece-2720-final-programming-competition
```

3. Extract the data:
```bash
unzip train.zip -d kaggle_upload/train
unzip test.zip -d kaggle_upload/test
# Place train.csv in kaggle_upload/
```

## Usage

Run the main script:
```bash
python bird_classifier.py
```

The script will:
1. Load training data and extract features
2. Split data into train/validation sets
3. Train multiple models and select the best one
4. Make predictions on the test set
5. Save predictions to `submission.csv`

## Model Training

The script trains multiple models and selects the best performing one based on validation accuracy:
- **SVM**: Tunes C parameter (0.1, 1, 10, 100, 1000)
- **KNN**: Tunes k parameter (3, 5, 7, 9, 11, 15, 20, 25)
- **Random Forest**: Uses 200 estimators with max_depth=30
- **Ensemble**: Voting classifier combining all three models

## Feature Extraction Details

The feature extraction pipeline includes:

1. **Spectral Features** (27 features):
   - FFT magnitude statistics (mean, median, std, min, max, percentiles, skew, kurtosis)
   - Spectral centroid, rolloff, bandwidth
   - Dominant frequency
   - Energy ratios in 5 frequency bands
   - Phase statistics

2. **Temporal Features** (10 features):
   - Amplitude statistics
   - Zero crossing rate
   - RMS energy
   - Autocorrelation features

3. **MFCC-like Features** (13 features):
   - Approximated MFCC coefficients using FFT and mel-scale filter bank

**Total: 50 features per audio sample**

## Output

- `submission.csv`: Predictions in the required format for Kaggle submission
- `model_*.pkl`: Saved models for each type (can be loaded later)

## Notes

- No GPU or neural network frameworks are used (as per competition rules)
- All features are extracted using NumPy and SciPy only
- The solution is designed to be robust to noise and recording quality variations


