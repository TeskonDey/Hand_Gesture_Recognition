"""
Hand Gesture Recognition - Real-Time Webcam Application
Detects hand, extracts landmarks, and classifies gesture at 30+ FPS.
"""

import os
import time
import cv2
import torch
import numpy as np
import mediapipe as mp

from model import HandGestureClassifier, normalize_landmarks, GESTURE_LABELS

def main():
    model_path = "models/gesture_model.pth"
    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model not found at '{model_path}'.")
        print("Please train the model first by running: python train.py")
        return

    # 1. Load PyTorch Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(GESTURE_LABELS)
    model = HandGestureClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[INFO] Loaded model successfully on {device}!")

    # 2. Setup MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # 3. Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    prev_time = time.time()
    fps = 0

    print("\n" + "=" * 50)
    print("  REAL-TIME HAND GESTURE RECOGNITION ACTIVE")
    print("  Press 'q' in the camera window to exit.")
    print("=" * 50)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Empty frame from webcam.")
            break

        # Mirror flip for intuitive interaction
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        predicted_label = "No Hand Detected"
        confidence = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw skeleton landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Extract 21 (x, y) coordinates
                raw_coords = []
                x_coords = []
                y_coords = []
                for lm in hand_landmarks.landmark:
                    raw_coords.append([lm.x, lm.y])
                    x_coords.append(int(lm.x * w))
                    y_coords.append(int(lm.y * h))

                # Normalize features
                features = normalize_landmarks(raw_coords)
                input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

                # Inference
                with torch.no_grad():
                    logits = model(input_tensor)
                    probabilities = torch.softmax(logits, dim=1)[0]
                    pred_class = torch.argmax(probabilities).item()
                    confidence = probabilities[pred_class].item() * 100.0
                    predicted_label = GESTURE_LABELS.get(pred_class, f"Class {pred_class}")

                # Draw bounding box around hand
                x_min, x_max = max(0, min(x_coords) - 20), min(w, max(x_coords) + 20)
                y_min, y_max = max(0, min(y_coords) - 20), min(h, max(y_coords) + 20)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 127), 2)

                # Put label tag directly above hand
                tag = f"{predicted_label} ({confidence:.1f}%)"
                cv2.putText(frame, tag, (x_min, max(25, y_min - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 127), 2)

        # Overlay Dashboard HUD
        cv2.rectangle(frame, (15, 15), (380, 100), (30, 30, 30), -1)
        cv2.putText(frame, f"Prediction: {predicted_label}", (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {confidence:.1f}%", (25, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1)
        cv2.putText(frame, f"FPS: {int(fps)} | Press 'q' to Quit", (25, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow("Hand Gesture Recognition (Computer Vision)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("[INFO] Application closed cleanly.")

if __name__ == "__main__":
    main()
