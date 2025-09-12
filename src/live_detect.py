import cv2
import joblib
from .pose_extractor import PoseExtractor
from .feature_engineer import extract_features

MODEL_PATH = "data/models/posture_model.pkl"

def run_live_detection():
    model = joblib.load(MODEL_PATH)
    pose_extractor = PoseExtractor()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = pose_extractor.extract_landmarks(frame)
        features = extract_features(landmarks)

        if features is not None:
            pred = model.predict([features])[0]
            prob = model.predict_proba([features])[0].max()

            label = f"{'Good' if pred==1 else 'Bad'} Posture ({prob:.2f})"
            cv2.putText(frame, label, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if pred==1 else (0,0,255), 2)

        cv2.imshow("Posture Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_detection()
