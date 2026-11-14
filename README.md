# Real-Time Hand Gesture Recognition using MediaPipe & Deep Neural Networks

An end-to-end computer vision and deep learning system that recognizes hand gestures (0 to 5) in real-time from webcam video streams at 30+ FPS.

---

## 1. Project Overview
This project classifies hand poses into 6 discrete gesture categories:
* `0`: Fist (Closed Hand)
* `1`: One / Index Pointer
* `2`: Two / Peace Sign
* `3`: Three
* `4`: Four
* `5`: Five / Open Palm

The pipeline separates feature detection from classification:
1. **Landmark Extraction:** Extracts 21 3D hand keypoints $(x, y, z)$ using Google MediaPipe's lightweight BlazePalm detector.
2. **Geometric Normalization:** Transforms raw coordinates into translation-invariant and scale-invariant 42-dimensional feature vectors.
3. **Deep Learning Classification:** A 4-layer Multilayer Perceptron (MLP) with Batch Normalization and Dropout for robust classification.

---

## 2. Proposed Architectural Diagram

```
+--------------------------------------------------------------------------------+
|                             SYSTEM ARCHITECTURE PIPELINE                       |
+--------------------------------------------------------------------------------+
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │  Video Stream (Webcam 30 FPS) │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ Frame Acquisition & RGB Conv  │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │  MediaPipe Hand Keypoint Extr │
                       │    (21 3D Spatial Landmarks)  │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ Geometric Normalization Block │
                       │   • Shift origin to Wrist (0) │
                       │   • Scale by max distance     │
                       └───────────────┬───────────────┘
                                       │  (42 Feature Vector)
                                       ▼
                       ┌───────────────────────────────┐
                       │  HandGestureClassifier (DNN)  │
                       │   • Linear(42, 128) + BN + RL │
                       │   • Dropout(p=0.2)            │
                       │   • Linear(128, 64) + BN + RL │
                       │   • Dropout(p=0.2)            │
                       │   • Linear(64, 32) + ReLU     │
                       │   • Linear(32, 6)             │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ Softmax Output & HUD Display  │
                       │ (Class Name, Confidence, FPS) │
                       └───────────────────────────────┘
```

---

## 3. Mathematical Normalization

To make the recognition invariant to hand position and distance from camera:

1. **Translation Invariance (Wrist as Origin):**
   $$x'_i = x_i - x_0, \quad y'_i = y_i - y_0 \quad (\text{for } i = 0, \dots, 20)$$
   where $(x_0, y_0)$ is the wrist joint coordinate.

2. **Scale Invariance:**
   $$d_{\max} = \max_{i} \left( |x'_i|, |y'_i| \right)$$
   $$\hat{x}_i = \frac{x'_i}{d_{\max}}, \quad \hat{y}_i = \frac{y'_i}{d_{\max}}$$
   The resulting vector $\mathbf{v} \in [-1.0, 1.0]^{42}$ represents pure relative hand geometry.

---

## 4. Model Hyperparameters

| Hyperparameter | Value |
| :--- | :--- |
| **Input Dimension** | 42 (21 landmarks $\times 2$ coordinates) |
| **Hidden Layers** | 128 $\rightarrow$ 64 $\rightarrow$ 32 |
| **Activation Function** | ReLU (Rectified Linear Unit) |
| **Regularization** | Batch Normalization, Dropout ($p=0.2$), Weight Decay ($10^{-4}$) |
| **Loss Function** | Cross-Entropy Loss |
| **Optimizer** | Adam ($\beta_1=0.9, \beta_2=0.999$) |
| **Initial Learning Rate** | $0.001$ |
| **Batch Size** | 32 |
| **Epochs** | 30 |

---

## 5. Installation & Setup

1. **Clone repository:**
   ```bash
   git clone https://github.com/your-username/hand-gesture-recognition.git
   cd hand-gesture-recognition
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 6. How to Run

### Step 1: Generate or Collect Data
* Option A (Fast Instant Start): Generate synthetic kinematic samples:
  ```bash
  python generate_dataset.py
  ```
* Option B (Custom Webcam Collection):
  ```bash
  python collect_data.py
  ```

### Step 2: Train Model
Train the deep neural network and automatically generate evaluation curves and confusion matrix:
```bash
python train.py
```
Outputs are saved to `models/gesture_model.pth` and `results/`.

### Step 3: Launch Real-Time Recognition
Run the live webcam recognition:
```bash
python app_webcam.py
```
*Press `q` to quit the application window.*

---

## 7. Results
* **Overall Test Accuracy:** ~99%
* **Inference Latency:** < 5 ms per frame (> 45 FPS on standard CPU)
* Generated figures:
  * `results/training_curves.png` (Train vs Val Loss and Accuracy)
  * `results/confusion_matrix.png` (Class Confusion Matrix)
