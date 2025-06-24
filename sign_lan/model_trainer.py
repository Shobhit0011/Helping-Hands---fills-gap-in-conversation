import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import os
from config import FEATURES_PATH, LABELS_PATH, MODEL_PATH, INPUT_SHAPE, BATCH_SIZE, EPOCHS, VALIDATION_SPLIT, SIGNS

class ModelTrainer:
    def __init__(self):
        self.model = self.create_model()

    def create_model(self):
        """Creates and compiles the model."""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(INPUT_SHAPE,)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(len(SIGNS), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        """Train the model using the saved dataset."""
        if not os.path.exists(FEATURES_PATH) or not os.path.exists(LABELS_PATH):
            print("Dataset files not found. Please ensure data is collected and saved.")
            return None

        # Load dataset
        X = np.load(FEATURES_PATH)
        y = np.load(LABELS_PATH)

        # Shuffle and split dataset
        indices = np.random.permutation(len(X))
        split_idx = int(len(X) * (1 - VALIDATION_SPLIT))
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Train the model
        print("\nTraining model...")
        history = self.model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, y_val)
        )

        # Save the model
        self.model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
        return history
