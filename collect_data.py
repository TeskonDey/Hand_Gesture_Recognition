"""
Hand Gesture Recognition - Live Data Collector via Webcam
Records custom hand landmark samples using modern MediaPipe Tasks.
Controls:
  [0-5] : Switch target gesture class
  'r'   : Start/pause capturing samples for current class
  'q'   : Save & Quit
"""

import os
import time
import urllib.request
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from model import normalize_landmarks, GESTURE_LABELS

TASK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
TASK_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

def ensure_task_model():
    if not os.path.exists(TASK_FILE):
        print("[INFO] Downloading hand_landmarker.task...")
        urllib.request.urlretrieve(TASK_URL, TASK_FILE)

def draw_skeleton(frame, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 255, 255), -1)
    return pts

def run_collector(output_csv="dataset/gesture_data.csv", samples_to_capture=100):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    ensure_task_model()

    base_options = python.BaseOptions(model_asset_path=TASK_FILE)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    current_class = 0
    recording = False
    captured_count = 0
    records = []

    print("\n--- Hand Gesture Data Collector ---")
    print("Controls:")
    print("  [0-5] : Switch target gesture class")
    print("  [r]   : Start/pause capturing samples")
    print("  [q]   : Save & Quit\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        res = detector.detect(mp_image)

        hand_detected = False
        if res.hand_landmarks:
            hand_detected = True
            lms = res.hand_landmarks[0]
            draw_skeleton(frame, lms, w, h)
            raw_coords = [[lm.x, lm.y] for lm in lms]

            if recording and captured_count < samples_to_capture:
                norm_feat = normalize_landmarks(raw_coords)
                records.append(list(norm_feat) + [current_class])
                captured_count += 1
                if captured_count >= samples_to_capture:
                    recording = False
                    print(f"[INFO] Captured {samples_to_capture} samples for Class {current_class} ({GESTURE_LABELS[current_class]})")

        status_color = (0, 0, 255) if recording else (0, 255, 0)
        status_text = f"RECORDING ({captured_count}/{samples_to_capture})" if recording else "IDLE (Press 'r' to record)"

        cv2.rectangle(frame, (10, 10), (450, 110), (20, 20, 20), -1)
        cv2.putText(frame, f"Class: {current_class} - {GESTURE_LABELS[current_class]}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, "Keys: 0-5 switch | 'r' record | 'q' quit", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Hand Gesture Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif ord('0') <= key <= ord('5'):
            current_class = key - ord('0')
            captured_count = 0
            recording = False
            print(f"[INFO] Selected Class: {current_class} - {GESTURE_LABELS[current_class]}")
        elif key == ord('r'):
            if hand_detected:
                recording = not recording
                if recording:
                    captured_count = 0
            else:
                print("[WARNING] No hand detected! Show your hand to webcam first.")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    if records:
        cols = [f"{axis}_{i}" for i in range(21) for axis in ("x", "y")] + ["label"]
        new_df = pd.DataFrame(records, columns=cols)
        if os.path.exists(output_csv):
            existing_df = pd.read_csv(output_csv)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(output_csv, index=False)
            print(f"[SUCCESS] Appended {len(records)} samples. Total dataset: {len(combined_df)}")
        else:
            new_df.to_csv(output_csv, index=False)
            print(f"[SUCCESS] Saved {len(records)} samples to {output_csv}")

if __name__ == "__main__":
    run_collector()
