"""
Hand Gesture Recognition - Real-Time Webcam Application
Detects hand, extracts 21 landmarks via MediaPipe Tasks, and classifies gesture at 30+ FPS.
"""

import os
import time
import urllib.request
import cv2
import torch
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from model import HandGestureClassifier, normalize_landmarks, GESTURE_LABELS

# MediaPipe Hand Model details
TASK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
TASK_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Standard 21 Hand Landmark Bones
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Knuckle joints connecting fingers
    (5, 9), (9, 13), (13, 17)
]


def ensure_task_model():
    """Downloads the official MediaPipe Hand Landmarker task file if missing."""
    if not os.path.exists(TASK_FILE):
        print(f"[INFO] Downloading hand_landmarker.task model...")
        urllib.request.urlretrieve(TASK_URL, TASK_FILE)
        print(f"[INFO] Download complete: {TASK_FILE}")


def draw_hand_skeleton(frame, landmarks, w, h):
    """Draws hand skeleton lines and landmark points on frame."""
    pixel_pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    
    # Draw connections (bones)
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, pixel_pts[start_idx], pixel_pts[end_idx], (0, 220, 100), 2)
        
    # Draw joint points
    for idx, (px, py) in enumerate(pixel_pts):
        # Red for finger tips (4, 8, 12, 16, 20), yellow for others
        color = (0, 0, 255) if idx in [4, 8, 12, 16, 20] else (0, 255, 255)
        cv2.circle(frame, (px, py), 5, color, -1)
        cv2.circle(frame, (px, py), 6, (0, 0, 0), 1)
        
    return pixel_pts


def main():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "gesture_model.pth")
    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model not found at '{model_path}'.")
        print("Please train the model first by running: python train.py")
        return

    # 1. Ensure MediaPipe model file exists
    ensure_task_model()

    # 2. Load PyTorch Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(GESTURE_LABELS)
    model = HandGestureClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[INFO] PyTorch model loaded successfully on {device}!")

    # 3. Setup Modern MediaPipe HandLandmarker
    base_options = python.BaseOptions(model_asset_path=TASK_FILE)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # 4. Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Please verify camera is connected.")
        return

    prev_time = time.time()
    fps = 0

    print("\n" + "=" * 55)
    print("      REAL-TIME HAND GESTURE RECOGNITION ACTIVE")
    print("      Show your hand to the camera.")
    print("      Press 'q' in the window to quit.")
    print("=" * 55 + "\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror frame horizontally for intuitive interaction
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        # Convert to RGB and process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        predicted_label = "No Hand Detected"
        confidence = 0.0

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Draw skeleton
                pixel_pts = draw_hand_skeleton(frame, hand_landmarks, w, h)

                # Extract normalized 42-D feature vector
                raw_coords = [[lm.x, lm.y] for lm in hand_landmarks]
                features = normalize_landmarks(raw_coords)
                input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

                # Run Neural Network Inference
                with torch.no_grad():
                    logits = model(input_tensor)
                    probabilities = torch.softmax(logits, dim=1)[0]
                    pred_class = torch.argmax(probabilities).item()
                    confidence = probabilities[pred_class].item() * 100.0
                    predicted_label = GESTURE_LABELS.get(pred_class, f"Class {pred_class}")

                # Draw bounding box
                xs = [pt[0] for pt in pixel_pts]
                ys = [pt[1] for pt in pixel_pts]
                x_min, x_max = max(0, min(xs) - 20), min(w, max(xs) + 20)
                y_min, y_max = max(0, min(ys) - 20), min(h, max(ys) + 20)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 127), 2)

                # Label badge above hand
                tag = f"{predicted_label} ({confidence:.1f}%)"
                cv2.putText(frame, tag, (x_min, max(25, y_min - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 127), 2)

        # Overlay Top HUD Dashboard
        cv2.rectangle(frame, (15, 15), (420, 105), (30, 30, 30), -1)
        cv2.putText(frame, f"Gesture: {predicted_label}", (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {confidence:.1f}%", (25, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1)
        cv2.putText(frame, f"FPS: {int(fps)} | Press 'q' to Quit", (25, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow("Hand Gesture Recognition (Computer Vision)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("[INFO] Application closed cleanly.")


if __name__ == "__main__":
    main()
