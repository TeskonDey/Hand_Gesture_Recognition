"""
Hand Gesture Recognition - Synthetic/Kinematic Dataset Generator
Generates realistic normalized 21-landmark coordinates for 6 gestures
with kinematic noise and rotational augmentation.
"""

import os
import numpy as np
import pandas as pd
from model import normalize_landmarks

# 21 Hand Landmarks Reference:
# 0: Wrist
# 1-4: Thumb (CMC, MCP, IP, Tip)
# 5-8: Index (MCP, PIP, DIP, Tip)
# 9-12: Middle (MCP, PIP, DIP, Tip)
# 13-16: Ring (MCP, PIP, DIP, Tip)
# 17-20: Pinky (MCP, PIP, DIP, Tip)

def get_base_skeleton():
    """Returns baseline 21 (x, y) coordinates for an open hand."""
    coords = [
        [0.50, 0.85], # 0: Wrist
        # Thumb
        [0.42, 0.76], [0.36, 0.68], [0.32, 0.60], [0.28, 0.54],
        # Index
        [0.43, 0.55], [0.41, 0.43], [0.40, 0.35], [0.39, 0.26],
        # Middle
        [0.50, 0.53], [0.50, 0.40], [0.50, 0.31], [0.50, 0.22],
        # Ring
        [0.57, 0.55], [0.58, 0.44], [0.59, 0.36], [0.60, 0.28],
        # Pinky
        [0.63, 0.60], [0.66, 0.52], [0.68, 0.46], [0.70, 0.39]
    ]
    return np.array(coords, dtype=np.float32)

def fold_finger(coords, start_idx, fold_factor=0.6):
    """Simulates curling/folding a finger inwards towards palm."""
    mcp = coords[start_idx]
    coords[start_idx + 1] = mcp + (coords[start_idx + 1] - mcp) * 0.4 + [0, 0.08]
    coords[start_idx + 2] = mcp + (coords[start_idx + 2] - mcp) * 0.2 + [0, 0.12]
    coords[start_idx + 3] = mcp + [0, 0.10]
    return coords

def generate_gesture_skeleton(gesture_id):
    """
    Creates landmark skeleton based on gesture class:
    0: Fist (all fingers folded)
    1: One / Point (index open, rest folded)
    2: Two / Peace (index and middle open, rest folded)
    3: Three (index, middle, ring open)
    4: Four (all 4 fingers open, thumb folded)
    5: Five / Open Palm (all fingers open)
    """
    hand = get_base_skeleton().copy()
    
    if gesture_id == 0:  # Fist
        hand = fold_finger(hand, 1)   # Thumb
        hand = fold_finger(hand, 5)   # Index
        hand = fold_finger(hand, 9)   # Middle
        hand = fold_finger(hand, 13)  # Ring
        hand = fold_finger(hand, 17)  # Pinky
    elif gesture_id == 1:  # One (Index)
        hand = fold_finger(hand, 1)   # Thumb
        hand = fold_finger(hand, 9)   # Middle
        hand = fold_finger(hand, 13)  # Ring
        hand = fold_finger(hand, 17)  # Pinky
    elif gesture_id == 2:  # Two (Peace)
        hand = fold_finger(hand, 1)   # Thumb
        hand = fold_finger(hand, 13)  # Ring
        hand = fold_finger(hand, 17)  # Pinky
    elif gesture_id == 3:  # Three
        hand = fold_finger(hand, 1)   # Thumb
        hand = fold_finger(hand, 17)  # Pinky
    elif gesture_id == 4:  # Four
        hand = fold_finger(hand, 1)   # Thumb
    elif gesture_id == 5:  # Five / Open
        pass # All extended
        
    return hand

def augment_skeleton(hand, noise_level=0.012, angle_range=15):
    """Applies realistic jitter noise, 2D rotation, and scaling."""
    # 1. Random 2D Rotation around wrist (index 0)
    theta = np.radians(np.random.uniform(-angle_range, angle_range))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    
    origin = hand[0]
    rotated = np.dot(hand - origin, R) + origin
    
    # 2. Random Scale (hand distance from camera)
    scale = np.random.uniform(0.85, 1.15)
    scaled = (rotated - origin) * scale + origin
    
    # 3. Add Gaussian landmark noise
    noise = np.random.normal(0, noise_level, size=hand.shape)
    augmented = scaled + noise
    
    return augmented

def create_dataset(samples_per_class=400, output_csv="dataset/gesture_data.csv"):
    """Generates the full dataset with 6 classes."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    data_records = []
    
    for gesture_id in range(6):
        base = generate_gesture_skeleton(gesture_id)
        for _ in range(samples_per_class):
            aug_hand = augment_skeleton(base)
            normalized = normalize_landmarks(aug_hand)
            record = list(normalized) + [gesture_id]
            data_records.append(record)
            
    # Build column headers
    cols = []
    for i in range(21):
        cols.append(f"x_{i}")
        cols.append(f"y_{i}")
    cols.append("label")
    
    df = pd.DataFrame(data_records, columns=cols)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    
    print(f"[SUCCESS] Dataset created at: {output_csv}")
    print(f"Total samples: {len(df)} ({samples_per_class} per class across 6 classes)")
    print(f"Feature vector length: 42 (21 landmarks x 2 coordinates)")

if __name__ == "__main__":
    create_dataset()
