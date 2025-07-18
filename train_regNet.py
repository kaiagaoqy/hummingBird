from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from regNN import RegressionNN, MultiHeadRegressor
import numpy as np
from numpy import genfromtxt
import logging
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# ========== Setup ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set up logging
logging.basicConfig(
    filename=f'logs/training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ========== Data Preparation ==========
# Example data placeholders 
X = genfromtxt('data/y.csv', delimiter=',').astype(np.float32) # Load input features
y = genfromtxt('data/X.csv', delimiter=',').astype(np.float32)  # Load target values

# Z-score normalization (StandardScaler)
x_scaler = StandardScaler()
y_scaler = StandardScaler()
from scipy.stats.mstats import winsorize


X = np.log1p(X+1)
y = np.log1p(y+1)
# X = winsorize(X, limits=[0.001, 0.001], axis=0)
# X[:,[1,3]] = np.log(X[:,[1,3]]+1)  # Apply log transformation to specific columns
X = x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y)

# X = X[:,[0,2,4]]
# y = y[:,[18]]

# Define per-group index ranges
group_ranges = [
    (0, 6),    # group 1
    (6, 12),    # group 2
    (12, 18),   # group 3
    (18, 19)   # group 4
]

# Optional: set custom weights for each group
group_weights = [1,1,1,1]  # Start equally weighted


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# ========== DataLoader with Batching ==========
batch_size = 64
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ========== Training Setup ==========
# Initialize network, loss, and optimizer
model = RegressionNN(input_size=X.shape[1], output_size=y.shape[1]).to(device)
# log model architecture
logging.info(f"Model architecture: {model}")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
warmup_epochs = 300

# Learning rate scheduler: Reduce LR when validation loss plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.8, patience=100
)

# Early stopping parameters
patience = 500
best_val_loss = np.inf
trigger_times = 0
epochs = 1000

for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        # Check gradient magnitude per parameter
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_mean = param.grad.abs().mean().item()
                logging.info(f"Grad {name}: {grad_mean:.6f}")
        optimizer.step()
        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)

    # Validation
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            running_val_loss += loss.item()

    avg_val_loss = running_val_loss / len(val_loader)

    # Logging
    lr = optimizer.param_groups[0]['lr']
    log = f"Epoch {epoch+1:03d} | LR: {lr:.5f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
    logging.info(log)

    if epoch >= warmup_epochs:
            scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pt")
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            print(f"Early stopping at epoch {epoch+1}")
            break

print(log)
# # Load the best model
# model.load_state_dict(torch.load('best_model.pt'))