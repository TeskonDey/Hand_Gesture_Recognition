"""
Hand Gesture Recognition - Live Data Collector via Webcam
Allows recording custom hand landmark samples using your own webcam.
Controls:
  '0' to '5': Select target gesture class
  'r': Record 100 samples for the selected class
  'q': Quit and close webcam
"""

import os
import time
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from model import normalize_landmarks, GESTURE_LABELS

def run_collector(output_csv="dataset/gesture_data.csv", samples_to_capture=100):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not access webcam. Please check connection.")
        return
        
    current_class = 0
    recording = False
    captured_count = 0
    records = []
    
    print("\n--- Hand Gesture Data Collector ---")
    print("Controls:")
    print("  [0-5] : Switch target gesture class")
    print("  [r]   : Start/pause capturing 100 samples")
    print("  [q]   : Save & Quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1) # Mirror flip
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        hand_detected = False
        if results.multi_hand_landmarks:
            hand_detected = True
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            raw_coords = []
            for lm in hand_landmarks.landmark:
                raw_coords.append([lm.x, lm.y])
                
            if recording and captured_count < samples_to_capture:
                norm_feat = normalize_landmarks(raw_coords)
                records.append(list(norm_feat) + [current_class])
                captured_count += 1
                if captured_count >= samples_to_capture:
                    recording = False
                    print(f"[INFO] Captured {samples_to_capture} samples for Class {current_class} ({GESTURE_LABELS[current_class]})")
                    
        # Overlay UI Information
        status_color = (0, 0, 255) if recording else (0, 255, 0)
        status_text = f"RECORDING ({captured_count}/{samples_to_capture})" if recording else "IDLE (Press 'r' to record)"
        
        cv2.rectangle(frame, (10, 10), (450, 110), (20, 20, 20), -1)
        cv2.putText(frame, f"Class: {current_class} - {GESTURE_LABELS[current_class]}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, "Keys: 0-5 to switch | 'r' record | 'q' quit", (20, 100),
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
    hands.close()
    
    if records:
        cols = []
        for i in range(21):
            cols.append(f"x_{i}")
            cols.append(f"y_{i}")
        cols.append("label")
        
        new_df = pd.DataFrame(records, columns=cols)
        if os.path.exists(output_csv):
            existing_df = pd.read_csv(output_csv)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(output_csv, index=False)
            print(f"[SUCCESS] Appended {len(records)} samples. Total dataset size: {len(combined_df)}")
        else:
            new_df.to_csv(output_csv, index=False)
            print(f"[SUCCESS] Saved {len(records)} samples to {output_csv}")

if __name__ == "__main__":
    run_collector()
