import os
import cv2
import pandas as pd
from .pose_extractor import PoseExtractor
from .feature_engineer import extract_features


GOOD_DIR = "data/good"
BAD_DIR = "data/bad"
OUTPUT_CSV = "data/processed/posture_data.csv"

pose_extractor = PoseExtractor()
rows = []

def process_folder(folder, label):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is None:
            continue
        landmarks = pose_extractor.extract_landmarks(img)
        features = extract_features(landmarks)
        if features is not None:
            rows.append(list(features) + [label])

# Process good/bad images
process_folder(GOOD_DIR, 1)
process_folder(BAD_DIR, 0)

# Save CSV
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
columns = ["head_tilt_x","head_tilt_y","shoulder_slope","hip_alignment","spine_tilt","neck_angle","slouch_depth","head_forward_depth","left_arm_y_ratio","right_arm_y_ratio","label"]
df = pd.DataFrame(rows, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved dataset with {len(df)} samples → {OUTPUT_CSV}")
