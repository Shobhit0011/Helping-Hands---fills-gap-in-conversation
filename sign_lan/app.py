import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from hand_detector import HandDetector
from data_collector import DataCollector
from model_trainer import ModelTrainer
from predictor import SignPredictor
from config import *

# Add project directory to Python path
project_dir = Path(__file__).resolve().parent
if str(project_dir) not in sys.path:
    sys.path.append(str(project_dir))

def initialize_collector():
    """Initialize HandDetector and DataCollector."""
    detector = HandDetector(
        static_mode=False,
        max_hands=1,
        detection_confidence=0.5,
        tracking_confidence=0.5
    )
    collector = DataCollector(
        hand_detector=detector,
        num_samples=SAMPLES_PER_SIGN
    )
    return collector

class SignLanguageTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Translator")
        self.root.geometry("400x300")

        self.label = tk.Label(root, text="Bridgify : Sign Language Recognition System", font=("Helvetica", 16))
        self.label.pack(pady=20)

        self.collect_data_button = tk.Button(root, text="Collect Training Data", command=self.collect_data)
        self.collect_data_button.pack(pady=10)

        self.train_model_button = tk.Button(root, text="Train Model", command=self.train_model)
        self.train_model_button.pack(pady=10)

        self.run_prediction_button = tk.Button(root, text="Run Real-Time Prediction", command=self.run_prediction)
        self.run_prediction_button.pack(pady=10)

        self.exit_button = tk.Button(root, text="Exit", command=root.quit)
        self.exit_button.pack(pady=10)

    def collect_data(self):
        collector = initialize_collector()
        print("\nStarting data collection...")
        print("Available signs:", list(SIGNS.values()))

        for sign_name, class_id in SIGNS.items():
            print(f"\nCollecting data for sign '{sign_name}'")
            print(f"Class ID: {class_id}")
            print("Press 'c' to start collecting samples.")
            print("Press 'q' to skip this sign.")

            try:
                collector.collect_data(sign_name, class_id)
            except Exception as e:
                messagebox.showerror("Error", f"Error collecting data for {sign_name}: {str(e)}")
                continue

            input(f"Data collection for '{sign_name}' completed. Press Enter to continue...")

        # Save the collected data
        collector.save_data()
        messagebox.showinfo("Info", "All data collection completed and saved in the dataset folder!")

    def train_model(self):
        features_path = os.path.join(DATASET_DIR, 'features.npy')
        labels_path = os.path.join(DATASET_DIR, 'labels.npy')

        if not os.path.exists(features_path) or not os.path.exists(labels_path):
            messagebox.showwarning("Warning", "Dataset not found. Please collect data first.")
            return

        trainer = ModelTrainer()
        history = trainer.train()

        if history:
            messagebox.showinfo("Info", "Training completed successfully! Model saved in models folder.")

    def run_prediction(self):
        model_path = os.path.join(MODEL_DIR, 'sign_language_model.h5')

        if not os.path.exists(model_path):
            messagebox.showwarning("Warning", "No trained model found. Please train the model first.")
            return

        predictor = SignPredictor()
        predictor.run_prediction()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageTranslatorApp(root)
    root.mainloop()