import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from torch import tensor, float32, no_grad, save
from torch.optim import Adam
from torch.nn import Module, LSTM, Linear, MSELoss
from torch.nn.init import constant_, xavier_normal_
from torch.nn.utils import clip_grad_norm_

from datetime import datetime
from glob import glob
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
# 1. Load the data
file_paths = glob('modified_data/gemini_data_*')
file_paths.sort()

file_idx = 12

print(f'Opening file: {file_paths[file_idx]}')
data = pd.read_csv(file_paths[file_idx])


# data.fillna(method='ffill', inplace=True)
# data.fillna(method='bfill', inplace=True)
data = data.dropna()

# For simplicity, let's just use 'open', 'high', 'low', 'close', 'volume'. 
# You can add more features if needed.
# features = ['open', 'high', 'low', 'close', 'Volume USD']
# features = ['Volume USD', 'SMA', 'EMA', 'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low', 'VWAP', 'Percentage_Returns', 'Log_Returns']
features = ['RSI', 'MACD', 'Percentage_Returns', 'Log_Returns', 'Target']
prices = data[features].values

assert not np.isnan(prices).any(), "There are still NaN values in the data!"
assert not np.isinf(prices).any(), "There are infinite values in the data!"

# 2. Pre-process the data
# scaler = MinMaxScaler()
# prices = scaler.fit_transform(prices)

# Use past 50 hours to predict the next hour
seq_length = 50
X, y = [], []
for i in range(len(prices) - seq_length):
    X.append(prices[i:i+seq_length])
    y.append(prices[i+seq_length, features.index('Target')]) # We want to predict the 'close' value
    # y.append(prices[i+seq_length, features.index('close')]) # We want to predict the 'close' value
# print(X)
# print(y)
# exit()
X = np.array(X)
y = np.array(y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
split_idx = int(len(X) * 0.8) # 80% for training
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Convert to PyTorch tensors
X_train = tensor(X_train, dtype=float32)
y_train = tensor(y_train, dtype=float32)
X_test = tensor(X_test, dtype=float32)
y_test = tensor(y_test, dtype=float32)

# Create DataLoader objects
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 3. Define LSTM model
class StockPredictor(Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(StockPredictor, self).__init__()
        self.lstm = LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

model = StockPredictor(input_dim=len(features), hidden_dim=30, num_layers=1) # Changed input_dim
# model = StockPredictor(input_dim=len(features), hidden_dim=50, num_layers=2) # Changed input_dim
criterion = MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)
for name, param in model.named_parameters():
    if 'bias' in name:
        constant_(param, 0.0)
    elif 'weight' in name:
        xavier_normal_(param)

# 4. Training and evaluation
num_epochs = 500
all_predictions = []
all_actual = []
train_losses = []
test_losses = []

print(f'{datetime.now()}: Starting predictions...')
for epoch in range(num_epochs):
    train_loss = 0
    model.train()
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs.view(-1), y_batch)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        max_gradient = max(p.grad.data.abs().max() for p in model.parameters())
        # print("Max gradient:", max_gradient.item())

        clip_grad_norm_(model.parameters(), max_norm=1)
        max_gradient = max(p.grad.data.abs().max() for p in model.parameters())
        # print("Max gradient normalized:", max_gradient.item())
        optimizer.step()

    model.eval()
    test_loss = 0
    with no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            all_predictions.extend(outputs.view(-1).tolist())
            all_actual.extend(y_batch.tolist())
            test_loss += criterion(outputs.view(-1), y_batch).item()

    train_losses.append(train_loss/len(train_loader))
    test_losses.append(test_loss/len(test_loader))
    print(f'{datetime.now()}: End of training epoch: {epoch+1}/{num_epochs}')
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}")

# Output the first few predictions and actual values
print("\nFirst 10 predictions:")
print(all_predictions[:10])
print("\nFirst 10 actual values:")
print(all_actual[:10])

# 5. Save the trained model
save(model.state_dict(), "lstm_stock_model.pth")

# To load the model later:
# model.load_state_dict(torch.load("lstm_stock_model.pth"))
# model.eval()

# 6. Visualization
# Convert UNIX timestamp to readable datetime
# Checking if the timestamps are in milliseconds
def check_timestamp(ts):
    return len(str(ts)) == 13

# Converting milliseconds to seconds
data['unix'] = data['unix'].apply(lambda x: x//1000 if check_timestamp(x) else x)

# Proceed with the rest of the code
# dates = pd.to_datetime(data['unix'], unit='s').dt.strftime('%Y-%m')
data['date'] = pd.to_datetime(data['unix'], unit='s')
dates = data['date'][seq_length:].values  # Removing first 'seq_length' rows as they are used for features and won't have a corresponding y value

# 1. Plotting features over time
fig, axes = plt.subplots(nrows=len(features), figsize=(14, 4*len(features)), sharex=True)

for i, (ax, feature) in enumerate(zip(axes, features)):
    ax.plot(dates, data[feature][seq_length:], label=feature)
    if i == len(features) - 1:  # Only label the bottom-most x-axis
        ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title(f'Feature: {feature}')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# 2. Display target and predicted values for "close" column over time
test_dates = dates[-len(y_test):]

# Repeating test dates for number of epochs
all_dates = np.tile(test_dates, num_epochs)

plt.figure(figsize=(14, 4))
plt.plot(all_dates, all_actual, label="Actual Target", color="blue")
plt.plot(all_dates, all_predictions, label="Predicted Target", color="red", alpha=0.5)
plt.xlabel('Date')
plt.ylabel('Target')
plt.title('Actual vs Predicted Target')
plt.legend()
plt.grid(True)
plt.show()

# # 3. Plot entire history of "close" column
# plt.figure(figsize=(14, 4))
# plt.plot(data['date'], data['Target'], label='Target', color='green')
# plt.xlabel('Date')
# plt.ylabel('Target')
# plt.title('History of Target')
# plt.legend()
# plt.grid(True)
# plt.show()

# 4. Plot train and test loss over prediction number
epochs = list(range(1, num_epochs + 1))
plt.figure(figsize=(14, 4))
plt.plot(epochs, train_losses, label="Train Loss", color="blue")
plt.plot(epochs, test_losses, label="Test Loss", color="red")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()