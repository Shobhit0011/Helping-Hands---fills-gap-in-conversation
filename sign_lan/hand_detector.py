import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self, static_mode=False, max_hands=1, detection_confidence=0.5, tracking_confidence=0.5):
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def find_hands(self, img, draw=True):
        """
        Finds hands in an image and optionally draws the landmarks.
        Returns the image with landmarks.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
        
        return img
    
    def find_positions(self, img, hand_number=0):
        """
        Returns a list of [id, x, y, z] for each landmark of the specified hand.
        """
        landmarks_list = []
        
        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > hand_number:
                hand = self.results.multi_hand_landmarks[hand_number]
                for id, landmark in enumerate(hand.landmark):
                    height, width, _ = img.shape
                    x, y, z = landmark.x * width, landmark.y * height, landmark.z
                    landmarks_list.append([id, x, y, z])
        
        return landmarks_list
    
    def get_landmark_array(self, img, hand_number=0):
        """
        Returns landmarks as a flat numpy array of shape (63,) for 21 landmarks.
        """
        positions = self.find_positions(img, hand_number)
        if not positions:
            return None
            
        landmarks_array = np.array(positions)[:, 1:]  # Extract x, y, z
        return landmarks_array.flatten()  # Shape: (63,)
    
    def release(self):
        """Placeholder for releasing resources (if needed)."""
        pass
