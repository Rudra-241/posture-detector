import numpy as np

def angle_between(p1, p2):
    """Compute angle (in radians) between two points for tilt"""
    dy = p2[1] - p1[1]
    dx = p2[0] - p1[0]
    return np.arctan2(dy, dx)

def extract_features(landmarks):
    """
    Converts MediaPipe pose landmarks into alignment-based features.
    landmarks: list of (x, y, z) tuples (33 points)
    Returns np.array of features or None
    """
    if landmarks is None or len(landmarks) < 33:
        return None

    # Indices for key landmarks
    NOSE = 0
    LEFT_EAR, RIGHT_EAR = 7, 8
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_ELBOW, RIGHT_ELBOW = 13, 14

    # Midpoint of shoulders
    mid_shoulder = [
        (landmarks[LEFT_SHOULDER][0] + landmarks[RIGHT_SHOULDER][0]) / 2,
        (landmarks[LEFT_SHOULDER][1] + landmarks[RIGHT_SHOULDER][1]) / 2
    ]

    # Normalization factor: shoulder width
    shoulder_width = np.linalg.norm(
        np.array(landmarks[LEFT_SHOULDER][:2]) - np.array(landmarks[RIGHT_SHOULDER][:2])
    )
    if shoulder_width == 0:
        shoulder_width = 1e-6  # prevent division by zero

    # Features
    features = {}

    # 1. Head tilt (nose relative to shoulder midpoint)
    features["head_tilt_x"] = (landmarks[NOSE][0] - mid_shoulder[0]) / shoulder_width
    features["head_tilt_y"] = (landmarks[NOSE][1] - mid_shoulder[1]) / shoulder_width

    # 2. Shoulder slope
    features["shoulder_slope"] = (
        landmarks[LEFT_SHOULDER][1] - landmarks[RIGHT_SHOULDER][1]
    ) / shoulder_width

    # 3. Neck angle (nose → shoulder line midpoint)
    features["neck_angle"] = angle_between(landmarks[NOSE], mid_shoulder)

    # 4. Arm raise ratios
    features["left_arm_y_ratio"] = (
        (landmarks[LEFT_SHOULDER][1] - landmarks[LEFT_ELBOW][1]) / shoulder_width
    )
    features["right_arm_y_ratio"] = (
        (landmarks[RIGHT_SHOULDER][1] - landmarks[RIGHT_ELBOW][1]) / shoulder_width
    )

    return np.array(list(features.values()))
