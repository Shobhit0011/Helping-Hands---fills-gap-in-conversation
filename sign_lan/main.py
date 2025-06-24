import os
import sys
from pathlib import Path
import cv2
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

def main():
    while True:
        print("\nSign Language Recognition System")
        print("1. Collect training data")
        print("2. Train model")
        print("3. Run real-time prediction")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            collector = initialize_collector()
            # Collect data for each sign using SIGNS from config
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
                    print(f"Error collecting data for {sign_name}: {str(e)}")
                    continue
                
                input(f"Data collection for '{sign_name}' completed. Press Enter to continue...")

            # Save the collected data
            collector.save_data()
            print("\nAll data collection completed and saved in the dataset folder!")
            
        elif choice == '2':
            features_path = os.path.join(DATASET_DIR, 'features.npy')
            labels_path = os.path.join(DATASET_DIR, 'labels.npy')
            
            if not os.path.exists(features_path) or not os.path.exists(labels_path):
                print("Dataset not found. Please collect data first.")
                continue
            
            trainer = ModelTrainer()
            history = trainer.train()
            
            if history:
                print("\nTraining completed successfully! Model saved in models folder.")
                
        elif choice == '3':
            model_path = os.path.join(MODEL_DIR, 'sign_language_model.h5')
            
            if not os.path.exists(model_path):
                print("No trained model found. Please train the model first.")
                continue
                
            predictor = SignPredictor()
            predictor.run_prediction()
            
        elif choice == '4':
            print("\nExiting...")
            break
            
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()
