# # # import cv2
# # # import joblib
# # # import mediapipe as mp
# # # from .pose_extractor import PoseExtractor
# # # from .feature_engineer import extract_features

# # # MODEL_PATH = "data/models/posture_model.pkl"

# # # # Mediapipe helpers
# # # mp_drawing = mp.solutions.drawing_utils
# # # mp_pose = mp.solutions.pose

# # # def run_live_detection():
# # #     model = joblib.load(MODEL_PATH)
# # #     pose_extractor = PoseExtractor()

# # #     cap = cv2.VideoCapture(0)

# # #     while cap.isOpened():
# # #         ret, frame = cap.read()
# # #         if not ret:
# # #             break

# # #         # Flip horizontally for a mirror view (optional)
# # #         frame = cv2.flip(frame, 1)

# # #         # Extract landmarks
# # #         landmarks, results = pose_extractor.extract_landmarks(frame, return_results=True)

# # #         # Draw pose landmarks (skeleton + points)
# # #         if results.pose_landmarks:
# # #             mp_drawing.draw_landmarks(
# # #                 frame,
# # #                 results.pose_landmarks,
# # #                 mp_pose.POSE_CONNECTIONS,
# # #                 mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
# # #                 mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
# # #             )

# # #             # Draw text labels for each landmark (x,y on frame)
# # #             h, w, _ = frame.shape
# # #             for idx, lm in enumerate(results.pose_landmarks.landmark):
# # #                 cx, cy = int(lm.x * w), int(lm.y * h)
# # #                 cv2.putText(frame, str(idx), (cx, cy - 5),
# # #                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# # #         # Run prediction if we have features
# # #         if landmarks is not None:
# # #             features = extract_features(landmarks)
# # #             if features is not None:
# # #                 pred = model.predict([features])[0]
# # #                 prob = model.predict_proba([features])[0].max()

# # #                 label = f"{'Good' if pred==1 else 'Bad'} Posture ({prob:.2f})"
# # #                 cv2.putText(frame, label, (30, 50),
# # #                             cv2.FONT_HERSHEY_SIMPLEX, 1,
# # #                             (0,255,0) if pred==1 else (0,0,255), 2)

# # #         cv2.imshow("Posture Detection", frame)
# # #         if cv2.waitKey(1) & 0xFF == ord("q"):
# # #             break

# # #     cap.release()
# # #     cv2.destroyAllWindows()

# # # if __name__ == "__main__":
# # #     run_live_detection()

# # # import cv2
# # # import joblib
# # # import mediapipe as mp
# # # import math
# # # from .pose_extractor import PoseExtractor
# # # from .feature_engineer import extract_features

# # # MODEL_PATH = "data/models/posture_model.pkl"

# # # # Mediapipe helpers
# # # mp_drawing = mp.solutions.drawing_utils
# # # mp_pose = mp.solutions.pose

# # # def run_live_detection():
# # #     model = joblib.load(MODEL_PATH)
# # #     pose_extractor = PoseExtractor()

# # #     cap = cv2.VideoCapture(0)

# # #     while cap.isOpened():
# # #         ret, frame = cap.read()
# # #         if not ret:
# # #             break

# # #         # Flip horizontally for a mirror view
# # #         frame = cv2.flip(frame, 1)

# # #         # Extract landmarks
# # #         landmarks, results = pose_extractor.extract_landmarks(frame, return_results=True)

# # #         debug_text = "No landmarks"

# # #         if results and results.pose_landmarks:
# # #             mp_drawing.draw_landmarks(
# # #                 frame,
# # #                 results.pose_landmarks,
# # #                 mp_pose.POSE_CONNECTIONS,
# # #                 mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
# # #                 mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
# # #             )

# # #             h, w, _ = frame.shape

# # #             def to_xy(idx):
# # #                 lm = results.pose_landmarks.landmark[idx]
# # #                 return int(lm.x * w), int(lm.y * h)

# # #             try:
# # #                 # Distance between midpoints of line 12–11 and line 10–9
# # #                 p11, p12 = to_xy(11), to_xy(12)
# # #                 p9, p10 = to_xy(9), to_xy(10)
# # #                 mid_shoulders = ((p11[0] + p12[0]) / 2, (p11[1] + p12[1]) / 2)
# # #                 mid_eyes = ((p9[0] + p10[0]) / 2, (p9[1] + p10[1]) / 2)
# # #                 dist_lines = math.dist(mid_shoulders, mid_eyes)

# # #                 # Angle line 4–8 from x-axis
# # #                 p4, p8 = to_xy(4), to_xy(8)
# # #                 angle_48 = math.degrees(math.atan2(p8[1] - p4[1], p8[0] - p4[0]))

# # #                 # Angle line 1–7 from x-axis
# # #                 p1, p7 = to_xy(1), to_xy(7)
# # #                 angle_17 = math.degrees(math.atan2(p7[1] - p1[1], p7[0] - p1[0]))

# # #                 debug_text = f"D: {dist_lines:.1f} | A(4-8): {angle_48:.1f}° | A(1-7): {angle_17:.1f}°"

# # #                 # Print in terminal too
# # #                 print(debug_text)

# # #             except Exception as e:
# # #                 debug_text = f"Error: {e}"
# # #                 print(debug_text)

# # #         # Overlay debug info
# # #         cv2.putText(frame, debug_text, (30, 90),
# # #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# # #         # Run prediction if we have features
# # #         if landmarks is not None:
# # #             features = extract_features(landmarks)
# # #             if features is not None:
# # #                 pred = model.predict([features])[0]
# # #                 prob = model.predict_proba([features])[0].max()

# # #                 label = f"{'Good' if pred==1 else 'Bad'} Posture ({prob:.2f})"
# # #                 cv2.putText(frame, label, (30, 50),
# # #                             cv2.FONT_HERSHEY_SIMPLEX, 1,
# # #                             (0,255,0) if pred==1 else (0,0,255), 2)

# # #         cv2.imshow("Posture Detection", frame)
# # #         if cv2.waitKey(1) & 0xFF == ord("q"):
# # #             break

# # #     cap.release()
# # #     cv2.destroyAllWindows()

# # # if __name__ == "__main__":
# # #     run_live_detection()



# # import cv2
# # import joblib
# # import mediapipe as mp
# # import math
# # from .pose_extractor import PoseExtractor
# # from .feature_engineer import extract_features

# # MODEL_PATH = "data/models/posture_model.pkl"

# # # Toggle: True = Rule-based | False = Model
# # USE_RULE_BASED = True

# # # Mediapipe helpers
# # mp_drawing = mp.solutions.drawing_utils
# # mp_pose = mp.solutions.pose

# # def run_live_detection():
# #     model = None
# #     if not USE_RULE_BASED:
# #         model = joblib.load(MODEL_PATH)

# #     pose_extractor = PoseExtractor()
# #     cap = cv2.VideoCapture(0)

# #     while cap.isOpened():
# #         ret, frame = cap.read()
# #         if not ret:
# #             break

# #         frame = cv2.flip(frame, 1)

# #         # Extract landmarks
# #         landmarks, results = pose_extractor.extract_landmarks(frame, return_results=True)

# #         debug_text = "No landmarks"
# #         label = ""

# #         if results and results.pose_landmarks:
# #             mp_drawing.draw_landmarks(
# #                 frame,
# #                 results.pose_landmarks,
# #                 mp_pose.POSE_CONNECTIONS,
# #                 mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
# #                 mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
# #             )

# #             h, w, _ = frame.shape

# #             def to_xy(idx):
# #                 lm = results.pose_landmarks.landmark[idx]
# #                 return int(lm.x * w), int(lm.y * h)

# #             try:
# #                 # Distance between midpoints of line 12–11 and line 10–9
# #                 p11, p12 = to_xy(11), to_xy(12)
# #                 p9, p10 = to_xy(9), to_xy(10)
# #                 mid_shoulders = ((p11[0] + p12[0]) / 2, (p11[1] + p12[1]) / 2)
# #                 mid_eyes = ((p9[0] + p10[0]) / 2, (p9[1] + p10[1]) / 2)
# #                 dist_lines = math.dist(mid_shoulders, mid_eyes)

# #                 # Angle line 4–8 from x-axis
# #                 p4, p8 = to_xy(4), to_xy(8)
# #                 angle_48 = math.degrees(math.atan2(p8[1] - p4[1], p8[0] - p4[0]))

# #                 # Angle line 1–7 from x-axis
# #                 p1, p7 = to_xy(1), to_xy(7)
# #                 angle_17 = math.degrees(math.atan2(p7[1] - p1[1], p7[0] - p1[0]))

# #                 debug_text = f"D: {dist_lines:.1f} | A(4-8): {angle_48:.1f}° | A(1-7): {angle_17:.1f}°"
# #                 print(debug_text)

# #                 if USE_RULE_BASED:
# #                     # --- Rule-based thresholds (tweak as needed) ---
# #                     if (110 < dist_lines < 160) and (140 < angle_48 < 160) and (20 < angle_17 < 30):
# #                         label = "Good Posture (Rule)"
# #                     else:
# #                         label = "Bad Posture (Rule)"

# #             except Exception as e:
# #                 debug_text = f"Error: {e}"
# #                 print(debug_text)

# #         # If ML model mode is active
# #         if not USE_RULE_BASED and landmarks is not None:
# #             features = extract_features(landmarks)
# #             if features is not None:
# #                 pred = model.predict([features])[0]
# #                 prob = model.predict_proba([features])[0].max()
# #                 label = f"{'Good' if pred==1 else 'Bad'} Posture ({prob:.2f})"

# #         # Overlay info
# #         if label:
# #             cv2.putText(frame, label, (30, 50),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 1,
# #                         (0,255,0) if "Good" in label else (0,0,255), 2)

# #         cv2.putText(frame, debug_text, (30, 90),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

# #         cv2.imshow("Posture Detection", frame)
# #         if cv2.waitKey(1) & 0xFF == ord("q"):
# #             break

# #     cap.release()
# #     cv2.destroyAllWindows()

# # if __name__ == "__main__":
# #     run_live_detection()


# import cv2
# import joblib
# import mediapipe as mp
# import math
# from .pose_extractor import PoseExtractor
# from .feature_engineer import extract_features

# MODEL_PATH = "data/models/posture_model.pkl"

# # Toggles
# USE_RULE_BASED = True   # True = Rule-based | False = Model
# DRAW_OVERLAY = False     # True = Draw landmarks + labels | False = Clean feed

# # Mediapipe helpers
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# def run_live_detection():
#     model = None
#     if not USE_RULE_BASED:
#         model = joblib.load(MODEL_PATH)

#     pose_extractor = PoseExtractor()
#     cap = cv2.VideoCapture(0)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.flip(frame, 1)
#         landmarks, results = pose_extractor.extract_landmarks(frame, return_results=True)

#         debug_text = "No landmarks"
#         label = ""

#         if results and results.pose_landmarks:
#             if DRAW_OVERLAY:
#                 mp_drawing.draw_landmarks(
#                     frame,
#                     results.pose_landmarks,
#                     mp_pose.POSE_CONNECTIONS,
#                     mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
#                     mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
#                 )

#             h, w, _ = frame.shape

#             def to_xy(idx):
#                 lm = results.pose_landmarks.landmark[idx]
#                 return int(lm.x * w), int(lm.y * h)

#             try:
#                 # Distance between midpoints of line 12–11 and line 10–9
#                 p11, p12 = to_xy(11), to_xy(12)
#                 p9, p10 = to_xy(9), to_xy(10)
#                 mid_shoulders = ((p11[0] + p12[0]) / 2, (p11[1] + p12[1]) / 2)
#                 mid_eyes = ((p9[0] + p10[0]) / 2, (p9[1] + p10[1]) / 2)
#                 dist_lines = math.dist(mid_shoulders, mid_eyes)

#                 # Angle line 4–8 from x-axis
#                 p4, p8 = to_xy(4), to_xy(8)
#                 angle_48 = math.degrees(math.atan2(p8[1] - p4[1], p8[0] - p4[0]))

#                 # Angle line 1–7 from x-axis
#                 p1, p7 = to_xy(1), to_xy(7)
#                 angle_17 = math.degrees(math.atan2(p7[1] - p1[1], p7[0] - p1[0]))

#                 debug_text = f"D: {dist_lines:.1f} | A(4-8): {angle_48:.1f}° | A(1-7): {angle_17:.1f}°"
#                 print(debug_text)

#                 if USE_RULE_BASED:
#                     # --- Rule-based thresholds ---
#                     if (120 < dist_lines < 150) and (140 < angle_48 < 160) and (20 < angle_17 < 30):
#                         label = "Good Posture (Rule)"
#                     else:
#                         label = "Bad Posture (Rule)"

#             except Exception as e:
#                 debug_text = f"Error: {e}"
#                 print(debug_text)

#         # If ML model mode is active
#         if not USE_RULE_BASED and landmarks is not None:
#             features = extract_features(landmarks)
#             if features is not None:
#                 pred = model.predict([features])[0]
#                 prob = model.predict_proba([features])[0].max()
#                 label = f"{'Good' if pred==1 else 'Bad'} Posture ({prob:.2f})"

#         # Overlay info only if enabled
#         if DRAW_OVERLAY:
#             if label:
#                 cv2.putText(frame, label, (30, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1,
#                             (0,255,0) if "Good" in label else (0,0,255), 2)

#             cv2.putText(frame, debug_text, (30, 90),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

#         cv2.imshow("Posture Detection", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     run_live_detection()


import cv2
import joblib
import mediapipe as mp
import math
from .pose_extractor import PoseExtractor
from .feature_engineer import extract_features

MODEL_PATH = "data/models/posture_model.pkl"

# Toggles
USE_RULE_BASED = True   # True = Rule-based | False = Model
DRAW_OVERLAY = False     # True = Show skeleton + debug | False = Only posture label

# Mediapipe helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def run_live_detection():
    model = None
    if not USE_RULE_BASED:
        model = joblib.load(MODEL_PATH)

    pose_extractor = PoseExtractor()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        landmarks, results = pose_extractor.extract_landmarks(frame, return_results=True)

        debug_text = "No landmarks"
        label = ""

        if results and results.pose_landmarks:
            if DRAW_OVERLAY:
                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                )

            h, w, _ = frame.shape

            def to_xy(idx):
                lm = results.pose_landmarks.landmark[idx]
                return int(lm.x * w), int(lm.y * h)

            try:
                # Distance between midpoints of line 12–11 and line 10–9
                p11, p12 = to_xy(11), to_xy(12)
                p9, p10 = to_xy(9), to_xy(10)
                mid_shoulders = ((p11[0] + p12[0]) / 2, (p11[1] + p12[1]) / 2)
                mid_eyes = ((p9[0] + p10[0]) / 2, (p9[1] + p10[1]) / 2)
                dist_lines = math.dist(mid_shoulders, mid_eyes)

                # Angle line 4–8 from x-axis
                p4, p8 = to_xy(4), to_xy(8)
                angle_48 = math.degrees(math.atan2(p8[1] - p4[1], p8[0] - p4[0]))

                # Angle line 1–7 from x-axis
                p1, p7 = to_xy(1), to_xy(7)
                angle_17 = math.degrees(math.atan2(p7[1] - p1[1], p7[0] - p1[0]))

                debug_text = f"D: {dist_lines:.1f} | A(4-8): {angle_48:.1f}° | A(1-7): {angle_17:.1f}°"
                print(debug_text)

                if USE_RULE_BASED:
                    # --- Rule-based thresholds ---
                    if (120 < dist_lines < 190) and (140 < angle_48 < 160) and (20 < angle_17 < 38):
                        label = "Good Posture (Rule)"
                    else:
                        label = "Bad Posture (Rule)"

            except Exception as e:
                debug_text = f"Error: {e}"
                print(debug_text)

        # If ML model mode is active
        if not USE_RULE_BASED and landmarks is not None:
            features = extract_features(landmarks)
            if features is not None:
                pred = model.predict([features])[0]
                prob = model.predict_proba([features])[0].max()
                label = f"{'Good' if pred==1 else 'Bad'} Posture ({prob:.2f})"

        # Always show posture label
        if label:
            cv2.putText(frame, label, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,255,0) if "Good" in label else (0,0,255), 2)

        # Only show debug text if overlay enabled
        if DRAW_OVERLAY:
            cv2.putText(frame, debug_text, (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Posture Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_detection()
