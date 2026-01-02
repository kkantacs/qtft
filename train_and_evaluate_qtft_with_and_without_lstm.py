import torch
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml

from qtft import QuantumMyModel 

# --- Quantile loss ---
def quantile_loss(y_pred, y_true, q=0.5):
    error = y_true - y_pred
    return torch.max((q - 1) * error, q * error).mean()

# --- Load data ---
# We demonstrate the model using Axis Bank stock data; a similar procedure
# is followed for the Seattle weather dataset.
# Files: "AXISBANK.csv" and "seattle-weather.csv"

df = pd.read_csv("AXISBANK.csv")
cols = ['Open', 'High', 'Low', 'Prev Close', 'Last', 'VWAP']
target_col = 'Close'
full_X = torch.tensor(df[cols].values, dtype=torch.float32)
full_y = torch.tensor(df[target_col].values, dtype=torch.float32).unsqueeze(-1)

# --- Hyperparameters ---
k, T, q = 2, 2, 0.5
learning_rate = 0.1
epoch_losses = []

# --- Initialize model ---
model = QuantumMyModel(num_vars_k=6, num_vars_T=6)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- Helper for prediction ---
def run_prediction(model, X, y, k, T):
    true_vals, pred_vals = [], []
    model.eval()
    with torch.no_grad():
        for i in range(20 - (k + T) + 1):
            window = X[i:i+k+T]
            window_y = y[i:i+k+T]
            x_k = window[:k].unsqueeze(0)
            x_T = window[k:k+T].unsqueeze(0)
            y_T = window_y[k:k+T].unsqueeze(0)

            y_pred = model(x_k, x_T)
            true_vals.extend(y_T.squeeze().tolist())
            pred_vals.extend(y_pred.squeeze().tolist())
    return true_vals, pred_vals

# --- Training loop ---
for epoch in range(101):
    model.train()
    total_loss = 0.0
    for i in range(20 - (k + T) + 1):
        window = full_X[i:i+k+T]
        window_y = full_y[i:i+k+T]
        x_k = window[:k].unsqueeze(0)
        x_T = window[k:k+T].unsqueeze(0)
        y_T = window_y[k:k+T].unsqueeze(0)

        y_pred = model(x_k, x_T)
        loss = quantile_loss(y_pred, y_T, q=q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epoch_losses.append(total_loss)

    # Visualize at key epochs
    if epoch in [0, 15, 30, 100]:
        print(f"Epoch {epoch} - Loss: {total_loss:.4f}")
        true_vals, pred_vals = run_prediction(model, full_X, full_y, k, T)
        plt.figure(figsize=(10, 4))
        plt.plot(true_vals, label='True', marker='o')
        plt.plot(pred_vals, label='Predicted', marker='x')
        plt.title(f"Epoch {epoch} - True vs Predicted")
        plt.xlabel("Time")
        plt.ylabel("Close Price")
        plt.legend()
        plt.grid(True)
        plt.show()

# --- Final loss plot ---
plt.figure(figsize=(10, 5))
plt.plot(epoch_losses, label='Training Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.grid(True)
plt.show()

# --- Test model ---
test_start, test_end = 18, 26
test_loss, test_count = 0.0, 0

model.eval()
with torch.no_grad():
    for i in range(test_start, test_end - (k + T) + 1):
        window = full_X[i:i+k+T]
        window_y = full_y[i:i+k+T]
        x_k = window[:k].unsqueeze(0)
        x_T = window[k:k+T].unsqueeze(0)
        y_T = window_y[k:k+T].unsqueeze(0)

        y_pred = model(x_k, x_T)
        loss = quantile_loss(y_pred, y_T, q=q)

        test_loss += loss.item()
        test_count += 1

avg_test_loss = test_loss / test_count if test_count > 0 else float('nan')
print(f"\nTest Loss (index {test_start} to {test_end}): {avg_test_loss:.4f}")
print(f"Test samples evaluated: {test_count}")

# --- Parameter count ---
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Trainable parameters: {count_parameters(model):,}")

