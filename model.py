"""
Hand Gesture Recognition - Model Architecture & Landmark Preprocessing
PyTorch implementation of the Deep Neural Network (MLP) Classifier.
"""

import torch
import torch.nn as nn
import numpy as np

# Label dictionary for the 6 gesture classes
GESTURE_LABELS = {
    0: "0 - Fist",
    1: "1 - Point / One",
    2: "2 - Peace / Two",
    3: "3 - Three",
    4: "4 - Four",
    5: "5 - Open Palm / High Five"
}


def normalize_landmarks(landmark_list):
    """
    Normalizes 21 (x, y) coordinates to achieve translation and scale invariance.
    
    Args:
        landmark_list: list of [x, y] coordinates for 21 hand landmarks.
        
    Returns:
        normalized_features: 42-dimensional numpy array of floats in [-1.0, 1.0].
    """
    landmarks = np.array(landmark_list, dtype=np.float32)  # Shape: (21, 2)
    
    # Step 1: Translation Invariance (Shift origin to wrist landmark #0)
    base_x, base_y = landmarks[0][0], landmarks[0][1]
    translated = landmarks - [base_x, base_y]
    
    # Flatten to 1D array of 42 values
    flattened = translated.flatten()
    
    # Step 2: Scale Invariance (Normalize by max coordinate distance)
    max_val = np.max(np.abs(flattened))
    if max_val > 1e-6:
        normalized = flattened / max_val
    else:
        normalized = flattened
        
    return normalized


class HandGestureClassifier(nn.Module):
    """
    Multilayer Perceptron (MLP) Neural Network for Landmark Classification.
    
    Architecture:
      Input (42) -> Linear(42, 128) -> BatchNorm -> ReLU -> Dropout(0.2)
                 -> Linear(128, 64) -> BatchNorm -> ReLU -> Dropout(0.2)
                 -> Linear(64, 32)  -> ReLU
                 -> Linear(32, num_classes) -> Logits
    """
    def __init__(self, num_classes=6, dropout_rate=0.2):
        super(HandGestureClassifier, self).__init__()
        
        self.network = nn.Sequential(
            # Block 1
            nn.Linear(42, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            # Block 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            # Block 3
            nn.Linear(64, 32),
            nn.ReLU(),
            
            # Output Classification Head
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    # Sanity check
    model = HandGestureClassifier(num_classes=6)
    dummy_input = torch.randn(8, 42)
    output = model(dummy_input)
    print("Model initialized successfully!")
    print(f"Input shape: {dummy_input.shape} -> Output shape: {output.shape}")
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
