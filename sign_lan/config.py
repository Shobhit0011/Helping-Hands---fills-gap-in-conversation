import os

# Base directory: Change to match your system
BASE_DIR = os.path.abspath("D:/Nicolas/sign_lan")  # Ensures an absolute path

# Dataset and model directories
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
FEATURES_PATH = os.path.join(DATASET_DIR, 'features.npy')
LABELS_PATH = os.path.join(DATASET_DIR, 'labels.npy')
SIGNS_DIR = os.path.join(DATASET_DIR, 'signs')

MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'sign_language_model.h5')

# Ensure required directories exist
directories = [DATASET_DIR, SIGNS_DIR, MODEL_DIR]
for directory in directories:
    try:
        os.makedirs(directory, exist_ok=True)  # Create if not exists
    except OSError as e:
        print(f"Error creating directory {directory}: {e}")
        raise

# Sign language alphabet mapping
SIGNS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
    6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
    12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z', 26: 'SPACE'
}

# Model parameters
INPUT_SHAPE = 63  # 21 landmarks × 3 coordinates
BATCH_SIZE = 32   # Number of samples per gradient update
EPOCHS = 50       # Number of training iterations
VALIDATION_SPLIT = 0.2  # Fraction of data for validation

# Data collection parameters
SAMPLES_PER_SIGN = 100  # Number of samples to collect per sign

# Confidence threshold for predictions
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to accept a prediction
