import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from torch import tensor, float32, no_grad, save, sigmoid
from torch.optim import Adam
from torch.nn import Module, LSTM, Linear, MSELoss, BCELoss
from torch.nn.init import constant_, xavier_normal_
from torch.nn.utils import clip_grad_norm_

from datetime import datetime
from glob import glob
from torch.utils.data import DataLoader, TensorDataset
import os
import json

class StockPredictor(Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(StockPredictor, self).__init__()
        self.lstm = LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5)
        self.linear = Linear(hidden_dim, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = sigmoid(self.linear(out[:, -1, :]))
        return out
    
TIME_DIFF_1 = 1
TIME_DIFF_24 = 24 # a day
TIME_DIFF_168 = 24 * 7 # a week
    
targets = [
    f"Target_shifted_{TIME_DIFF_1}",
    f"Target_shifted_{TIME_DIFF_24}",
    f"Target_shifted_{TIME_DIFF_168}"
]

file_idx = 12
num_epochs = 10
all_predictions = []
all_actual = []
train_losses = []
test_losses = []
# target_feature = "Target"
features = ['RSI', 'MACD', 'Percentage_Returns', 'Log_Returns'] + targets
seq_length = 50
X, y = [], []
model = StockPredictor(input_dim=len(features), hidden_dim=50, num_layers=2)
criterion = BCELoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

file_paths = glob('modified_data/gemini_data_*')
file_paths.sort()

print(f'Opening file: {file_paths[file_idx]}')
data = pd.read_csv(file_paths[file_idx])
data = data.dropna()
prices = data[features].values
assert not np.isnan(prices).any(), "There are still NaN values in the data!"
assert not np.isinf(prices).any(), "There are infinite values in the data!"

# for i in range(len(prices) - seq_length):
#     X.append(prices[i:i+seq_length])
#     y.append(prices[i+seq_length, features.index(target_feature)]) # We want to predict the target_feature value

for i in range(len(prices) - seq_length):
    X.append(prices[i:i+seq_length])
    y.append([prices[i+seq_length, features.index(target)] for target in targets])

X = np.array(X)
y = np.array(y)

split_idx = int(len(X) * 0.8) # 80% for training
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

X_train = tensor(X_train, dtype=float32)
y_train = tensor(y_train, dtype=float32).reshape(-1, 3)
X_test = tensor(X_test, dtype=float32)
y_test = tensor(y_test, dtype=float32).reshape(-1, 3)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for name, param in model.named_parameters():
    if 'bias' in name:
        constant_(param, 0.0)
    elif 'weight' in name:
        xavier_normal_(param)

for epoch in range(num_epochs):
    train_loss = 0
    model.train()
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    test_loss = 0
    with no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predicted_probs = outputs.tolist()  # Store the raw probability
            all_predictions.extend(predicted_probs)
            all_actual.extend(y_batch.tolist())
            test_loss += criterion(outputs, y_batch).item()

    train_losses.append(train_loss/len(train_loader))
    test_losses.append(test_loss/len(test_loader))
    print(f"{datetime.now()}: Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}")

# 5. Save the trained model
# save(model.state_dict(), "lstm_stock_model.pth")

def evaluate_threshold(actual, predicted, threshold):
    predicted_tensor = tensor(predicted)
    predicted_labels = (predicted_tensor > threshold).int().tolist()
    correct = sum([1 for act, pred in zip(actual, predicted_labels) if act == pred])
    accuracy = correct / len(actual)
    return accuracy


thresholds = [0.4, 0.5, 0.6]
for thresh in thresholds:
    accuracy = evaluate_threshold(all_actual, all_predictions, thresh)
    print(f"Accuracy for threshold {thresh}: {accuracy:.4f}")

# Note that all_actual and all_predictions should be lists of lists where 
# each inner list is of length 3 (since there are 3 predictions per entry)

# Initializing counts
actual_ones = [0, 0, 0]
actual_zeros = [0, 0, 0]
predicted_ones = [0, 0, 0]
predicted_zeros = [0, 0, 0]
false_positives = [0, 0, 0]
false_negatives = [0, 0, 0]

used_threshold = thresholds[-1]

for actual_values, predicted_values in zip(all_actual, all_predictions):
    for idx, (actual, predicted) in enumerate(zip(actual_values, predicted_values)):
        if actual == 1:
            actual_ones[idx] += 1
            if predicted < used_threshold:
                false_negatives[idx] += 1
        else:
            actual_zeros[idx] += 1
            if predicted >= used_threshold:
                false_positives[idx] += 1
        
        if predicted >= used_threshold:
            predicted_ones[idx] += 1
        else:
            predicted_zeros[idx] += 1

print(f'all_predictions: {all_predictions}')

print("\nSummary:")

targets = ["1 hour", "24 hours", "168 hours"]
for i, target in enumerate(targets):
    print(f"Metrics for prediction over {target}:")
    print(f"Actual 1s: {actual_ones[i]}")
    print(f"Actual 0s: {actual_zeros[i]}")
    print(f"Predicted 1s: {predicted_ones[i]}")
    print(f"Predicted 0s: {predicted_zeros[i]}")
    print(f"false positives: {false_positives[i]}")
    print(f"false negatives: {false_negatives[i]}")
    print(f"False Positive Ratio: {false_positives[i] / (false_positives[i] + actual_zeros[i]):.4f}")  
    print(f"False Negative Ratio: {false_negatives[i] / (false_negatives[i] + actual_ones[i]):.4f}")  
    print("------\n")


def check_timestamp(ts):
    return len(str(ts)) == 13

# Converting milliseconds to seconds
data['unix'] = data['unix'].apply(lambda x: x//1000 if check_timestamp(x) else x)
data['date'] = pd.to_datetime(data['unix'], unit='s')
dates = data['date'][seq_length:].values  # Removing first 'seq_length' rows as they are used for features and won't have a corresponding y value

if not os.path.exists('results'):
    os.makedirs('results')

# Save to a file
current_time = datetime.now().strftime('%Y%m%d_%H%M')
file_name = f"results/model_data_{current_time}.json"

model_data = {
    "Training File": file_paths[file_idx],
    "Features Used": features,
    "Target Feature": features,  # TODO Refactoring required
    "Sequence Length": seq_length,
    "Model Structure": str(model),
    "Loss Function": str(criterion),
    "Epochs": num_epochs,
    "Thresholds": thresholds,
    "Train Loss Over Epochs": train_losses,
    "Test Loss Over Epochs": test_losses
}

with open(file_name, 'w') as file:
    json.dump(model_data, file, indent=4)

print(f"Saved model data to {file_name}")

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

# plt.figure(figsize=(14, 4))
# plt.plot(all_dates, all_actual, label="Actual Target", color="blue")
# plt.plot(all_dates, all_predictions, label="Predicted Target", color="red", alpha=0.5)
# plt.xlabel('Date')
# plt.ylabel('Target')
# plt.title('Actual vs Predicted Target')
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