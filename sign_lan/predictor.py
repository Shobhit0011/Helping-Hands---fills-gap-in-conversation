import cv2
import numpy as np
import tensorflow as tf
from config import *
from hand_detector import HandDetector

class SignPredictor:
    def __init__(self):
        """Initialize the predictor with hand detector and model."""
        self.hand_detector = HandDetector()
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.current_word = []
        self.last_prediction = None
        self.prediction_count = 0

    def predict_sign(self, landmarks):
        """Predict the sign from hand landmarks."""
        if landmarks is None:
            return None, 0
        
        landmarks = landmarks.reshape(1, -1)
        prediction = self.model.predict(landmarks, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        return predicted_class, confidence

    def run_prediction(self):
        """Run real-time sign language prediction."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open the camera.")
            return
        
        cv2.namedWindow('Sign Language Prediction', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Sign Language Prediction', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        print("\nSign Language Prediction Started")
        print("Controls:")
        print("- Press SPACE to add a space")
        print("- Press BACKSPACE to delete last character")
        print("- Press ENTER to clear the text")
        print("- Press 'q' to quit")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break
            
            frame = cv2.flip(frame, 1)  # Mirror image
            frame = self.hand_detector.find_hands(frame)
            landmarks = self.hand_detector.get_landmark_array(frame)
            
            if landmarks is not None:
                predicted_class, confidence = self.predict_sign(landmarks)
                
                if confidence > CONFIDENCE_THRESHOLD:
                    current_prediction = SIGNS[predicted_class]
                    
                    if current_prediction == self.last_prediction:
                        self.prediction_count += 1
                    else:
                        self.prediction_count = 0
                    
                    if self.prediction_count == 10:  # Add letter after consistent prediction
                        self.current_word.append(current_prediction)
                        print(f"Letter added: {current_prediction}")
                        self.prediction_count = 0
                    
                    self.last_prediction = current_prediction
                    
                    # Display prediction
                    cv2.putText(frame, f"Predicted: {current_prediction} ({confidence:.2f})",
                              (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display current word
            current_text = ''.join(self.current_word)
            cv2.putText(frame, f"Word: {current_text}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            cv2.imshow('Sign Language Prediction', frame)
            key = cv2.waitKey(1)
            
            if key & 0xFF == ord('q'):
                break
            elif key == 32:  # SPACE
                self.current_word.append(' ')
                print("Space added")
            elif key == 8:  # BACKSPACE
                if self.current_word:
                    removed = self.current_word.pop()
                    print(f"Removed: {removed}")
            elif key == 13:  # ENTER
                self.current_word = []  # Clear the text
                print("Text cleared")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hand_detector.release()

if __name__ == "__main__":
    predictor = SignPredictor()
    predictor.run_prediction()