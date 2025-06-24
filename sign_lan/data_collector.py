# data_collector.py
import cv2
import numpy as np
from pathlib import Path
import os

class DataCollector:
    def __init__(self, hand_detector, num_samples=100):
        """
        Initialize the data collector with a hand detector instance
        and the number of samples to collect per sign.
        """
        self.hand_detector = hand_detector
        self.num_samples = num_samples
        self.dataset = []
        self.labels = []
        self.sign_data = {}  # Dictionary to store data for each sign
        
    def collect_data(self, sign_name, class_id):
        """
        Collect data for a specific sign.
        """
        cap = cv2.VideoCapture(0)
        collected_samples = 0
        collecting = False
        current_sign_data = []  # Store data for current sign
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Flip frame for selfie view
            frame = cv2.flip(frame, 1)
            
            # Find hands and draw landmarks
            frame = self.hand_detector.find_hands(frame)
            
            # Display instructions and status
            self._display_info(frame, sign_name, class_id, collected_samples)
            
            if collecting:
                # Get landmarks as flat array
                landmarks = self.hand_detector.get_landmark_array(frame)
                
                if landmarks is not None:
                    current_sign_data.append(landmarks)
                    self.dataset.append(landmarks)
                    self.labels.append(class_id)
                    collected_samples += 1
                    print(f"Collected sample {collected_samples}/{self.num_samples} for sign {sign_name}")
                    
                    if collected_samples >= self.num_samples:
                        # Store data for this sign
                        self.sign_data[sign_name] = {
                            'features': np.array(current_sign_data),
                            'label': class_id
                        }
                        break
            
            # Handle key presses
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('c'):
                collecting = True
                
            cv2.imshow('Data Collection', frame)
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _display_info(self, frame, sign_name, class_id, collected_samples):
        """
        Display information on the frame.
        """
        cv2.putText(frame, f"Sign '{sign_name}' (Class {class_id})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'c' to start collecting, 'q' to quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Collected: {collected_samples}/{self.num_samples}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def save_data(self, output_dir='dataset'):
        """
        Save collected data to numpy files.
        """
        if not self.dataset:
            print("No data to save!")
            return
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual sign data
        signs_dir = output_path / 'signs'
        signs_dir.mkdir(exist_ok=True)
        
        for sign_name, data in self.sign_data.items():
            sign_path = signs_dir / sign_name
            sign_path.mkdir(exist_ok=True)
            
            # Save features and label for this sign
            np.save(sign_path / 'features.npy', data['features'])
            np.save(sign_path / 'label.npy', data['label'])
            print(f"Saved {len(data['features'])} samples for sign {sign_name}")
        
        # Save complete dataset
        X = np.array(self.dataset)  # Shape: (n_samples, 63)
        y = np.array(self.labels)   # Shape: (n_samples,)
        
        # Save arrays
        np.save(output_path / 'features.npy', X)
        np.save(output_path / 'labels.npy', y)
        print(f"\nComplete dataset:")
        print(f"Total samples: {len(self.dataset)}")
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"\nData saved in:")
        print(f"- Complete dataset: {output_path}")
        print(f"- Individual signs: {signs_dir}")
        
    def get_data(self):
        """
        Return the collected data and labels.
        """
        return np.array(self.dataset), np.array(self.labels)