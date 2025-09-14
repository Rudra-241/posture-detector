import mediapipe as mp
import cv2

class PoseExtractor:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_landmarks(self, frame, return_results=False):
        """Extract pose landmarks from a frame.
        Args:
            frame: input BGR frame (OpenCV).
            return_results (bool): whether to also return raw MediaPipe results.
        Returns:
            - landmarks: list of (x, y, z) in image coordinates (or None).
            - results (optional): full MediaPipe results if return_results=True.
        """
        # Convert BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks:
            h, w, _ = frame.shape
            landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
        else:
            landmarks = None

        if return_results:
            return landmarks, results
        return landmarks
