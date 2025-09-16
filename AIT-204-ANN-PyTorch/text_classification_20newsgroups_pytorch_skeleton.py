# text_classification_20newsgroups_pytorch_completed.py
# Purpose: TF-IDF + PyTorch MLP for 20 Newsgroups

import os
import random
import numpy as np

# ---- Reproducibility (optional but recommended) ----
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

import torch
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# Load Dataset
# =========================
print("Loading 20 Newsgroups dataset...")
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
X_raw, y = data.data, data.target
num_classes = len(data.target_names)
print("Dataset loaded.")

# =========================
# Preprocess text data (Handled by TfidfVectorizer)
# =========================
# The TfidfVectorizer below handles lowercase conversion, stop word removal,
# and tokenization according to the specified token_pattern.

# =========================
# Convert Text Data to Numerical Format
# =========================
print("Vectorizing text data with TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=5000,          # Limit to 5000 most frequent words
    lowercase=True,
    stop_words='english',
    strip_accents='unicode',
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
)
X_vec = vectorizer.fit_transform(X_raw)
X_vec = X_vec.toarray()         # Densify the matrix
print("Vectorization complete.")

# =========================
# Split data
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, stratify=y, random_state=SEED
)

# =========================
# Torch Tensors & Dataloaders
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

from torch.utils.data import TensorDataset, DataLoader

train_ds = TensorDataset(X_train_t, y_train_t)
test_ds  = TensorDataset(X_test_t,  y_test_t)

# Batch size is a key hyperparameter. 64 is a reasonable default.
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, drop_last=False)

# =========================
# Design Neural Network Architecture
# =========================
import torch.nn as nn

class NewsMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # A simple but effective MLP structure for this task
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Forward pass through the defined network
        return self.net(x)

input_dim = X_train_t.shape[1]
model = NewsMLP(input_dim=input_dim, num_classes=num_classes).to(device)
print("\nModel Architecture:")
print(model)

# =========================
# Compile the Model (PyTorch-style setup)
# =========================
# Adam is a robust, general-purpose optimizer.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# CrossEntropyLoss is standard for multi-class classification.
criterion = nn.CrossEntropyLoss()
# An optional scheduler to reduce learning rate if training plateaus.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

# =========================
# Train the Model
# =========================
def train(num_epochs=10):
    """
    Implements the training loop for the PyTorch model.
    """
    assert optimizer is not None and criterion is not None, "Set optimizer/criterion before training."
    print("\n--- Starting Training ---")
    for epoch in range(num_epochs):
        model.train() # Set the model to training mode
        running_loss, running_correct, total = 0.0, 0, 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            # Standard training steps
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            # (optional) Calculate metrics for this batch
            preds = logits.argmax(dim=1)
            running_loss += loss.item() * xb.size(0)
            running_correct += (preds == yb).sum().item()
            total += xb.size(0)

        epoch_loss = running_loss / max(1, total)
        epoch_acc  = running_correct / max(1, total)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

        # Step the scheduler based on the training loss
        if scheduler is not None:
            scheduler.step(epoch_loss)
    print("--- Training Finished ---")

# =========================
# Evaluate the Model
# =========================
def evaluate():
    """
    Evaluates the model on the test dataset and prints classification metrics.
    """
    print("\n--- Starting Evaluation ---")
    model.eval() # Set the model to evaluation mode
    all_preds, all_targets = [], []
    with torch.no_grad(): # Disable gradient calculations for efficiency
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    # Concatenate results from all batches
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    # Print comprehensive evaluation metrics
    print(f"\nTest Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=data.target_names, zero_division=0))
    # print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    train(num_epochs=10)
    evaluate()