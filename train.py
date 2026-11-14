"""
Hand Gesture Recognition - Model Training & Evaluation Pipeline
Trains a PyTorch neural network, outputs learning curves, confusion matrix, and saves weights.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from model import HandGestureClassifier, GESTURE_LABELS


class GestureDataset(Dataset):
    """Custom PyTorch Dataset for Landmark coordinates."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_gesture_model(
    csv_path="dataset/gesture_data.csv",
    epochs=30,
    batch_size=32,
    lr=0.001,
    save_dir="models",
    results_dir="results"
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Run generate_dataset.py or collect_data.py first!")

    # 1. Load Data
    print(f"[INFO] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values # 42 features
    y = df.iloc[:, -1].values  # class labels
    
    num_classes = len(np.unique(y))
    print(f"[INFO] Loaded {len(df)} samples across {num_classes} classes.")
    
    # 2. Stratified Train / Validation Split (80% / 20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    train_loader = DataLoader(GestureDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(GestureDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    
    # 3. Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training device: {device}")
    
    model = HandGestureClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Tracking metrics
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }
    
    print("\n" + "=" * 60)
    print("           STARTING MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    best_val_acc = 0.0
    model_save_path = os.path.join(save_dir, "gesture_model.pth")
    
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
        train_loss = running_loss / total
        train_acc = (correct / total) * 100.0
        
        # Validation Phase
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
        val_loss = val_running_loss / val_total
        val_acc = (val_correct / val_total) * 100.0
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
                  
    print("=" * 60)
    print(f"[SUCCESS] Model training complete! Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"[SAVED] Best model checkpoint saved to: {model_save_path}")
    
    # 4. Final Evaluation & Classification Report
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    target_names = [GESTURE_LABELS.get(i, f"Class {i}") for i in range(num_classes)]
    
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(all_targets, all_preds, target_names=target_names))
    
    # 5. Generate and Save Result Plots
    # A) Loss and Accuracy Curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history["train_loss"], label="Train Loss", color="royalblue", lw=2)
    plt.plot(range(1, epochs + 1), history["val_loss"], label="Val Loss", color="crimson", lw=2, linestyle="--")
    plt.title("Loss Curves over Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history["train_acc"], label="Train Accuracy", color="royalblue", lw=2)
    plt.plot(range(1, epochs + 1), history["val_acc"], label="Val Accuracy", color="crimson", lw=2, linestyle="--")
    plt.title("Accuracy Curves over Epochs (%)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    curve_plot_path = os.path.join(results_dir, "training_curves.png")
    plt.savefig(curve_plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] Training curves saved to: {curve_plot_path}")
    
    # B) Confusion Matrix Heatmap
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"C{i}" for i in range(num_classes)],
                yticklabels=[f"C{i}" for i in range(num_classes)])
    plt.title("Confusion Matrix (Validation Set)", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.tight_layout()
    cm_plot_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] Confusion matrix saved to: {cm_plot_path}")


if __name__ == "__main__":
    train_gesture_model()
