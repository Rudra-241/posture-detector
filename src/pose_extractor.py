import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

class PoseExtractor:
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

    def extract_landmarks(self, frame):
        """Extract normalized landmarks from a frame"""
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None
        return [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
