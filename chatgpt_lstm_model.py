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

file_idx = 13
num_epochs = 10
all_predictions = []
all_actual = []
train_losses = []
test_losses = []
# target_feature = "Target"
features = ['RSI', 'MACD', 'Percentage_Returns', 'Log_Returns']
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

all_columns = features + targets
prices = data[all_columns].values

assert not np.isnan(prices).any(), "There are still NaN values in the data!"
assert not np.isinf(prices).any(), "There are infinite values in the data!"

X, y = [], []


# for i in range(len(prices) - seq_length):
#     X.append(prices[i:i+seq_length])
#     y.append(prices[i+seq_length, features.index(target_feature)]) # We want to predict the target_feature value

for i in range(len(prices) - seq_length):
    X.append(prices[i:i+seq_length, :len(features)])
    y_values = [prices[i+seq_length, len(features)+target_idx] for target_idx in range(len(targets))]
    y.append(y_values)

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

# thresholds = [0.4, 0.5, 0.6]

def compute_metrics_for_threshold(all_actual, all_predictions, threshold):
    actual_ones = [0, 0, 0]
    actual_zeros = [0, 0, 0]
    predicted_ones = [0, 0, 0]
    predicted_zeros = [0, 0, 0]
    false_positives = [0, 0, 0]
    false_negatives = [0, 0, 0]

    for actual_values, predicted_values in zip(all_actual, all_predictions):
        for idx, (actual, predicted) in enumerate(zip(actual_values, predicted_values)):
            if actual == 1:
                actual_ones[idx] += 1
                if predicted < threshold:
                    false_negatives[idx] += 1
            else:
                actual_zeros[idx] += 1
                if predicted >= threshold:
                    false_positives[idx] += 1

            if predicted >= threshold:
                predicted_ones[idx] += 1
            else:
                predicted_zeros[idx] += 1

    metrics = {
        "actual_ones": actual_ones,
        "actual_zeros": actual_zeros,
        "predicted_ones": predicted_ones,
        "predicted_zeros": predicted_zeros,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }
    return metrics


targets = ["1 hour", "24 hours", "168 hours"]
evaluation_metrics_by_threshold = []

for thresh in thresholds:
    print(f"\nMetrics for threshold {thresh}:")
    metrics = compute_metrics_for_threshold(all_actual, all_predictions, thresh)

    metrics_for_threshold = {
        "Threshold": thresh
    }
    for i, target in enumerate(targets):
        print(f"Metrics for prediction over {target}:")
        print(f"Actual 1s: {metrics['actual_ones'][i]}")
        print(f"Actual 0s: {metrics['actual_zeros'][i]}")
        print(f"Predicted 1s: {metrics['predicted_ones'][i]}")
        print(f"Predicted 0s: {metrics['predicted_zeros'][i]}")
        print(f"False Positives: {metrics['false_positives'][i]}")
        print(f"False Negatives: {metrics['false_negatives'][i]}")
        print(f"False Positive Ratio: {metrics['false_positives'][i] / (metrics['false_positives'][i] + metrics['actual_zeros'][i]):.4f}")
        print(f"False Negative Ratio: {metrics['false_negatives'][i] / (metrics['false_negatives'][i] + metrics['actual_ones'][i]):.4f}")
        print("------\n")
        # Storing metrics for this target
        metrics_for_threshold[target] = {
            "Actual 1s": metrics['actual_ones'][i],
            "Actual 0s": metrics['actual_zeros'][i],
            "Predicted 1s": metrics['predicted_ones'][i],
            "Predicted 0s": metrics['predicted_zeros'][i],
            "False Positives": metrics['false_positives'][i],
            "False Negatives": metrics['false_negatives'][i],
            "False Positive Ratio": metrics['false_positives'][i] / (metrics['false_positives'][i] + metrics['actual_zeros'][i]),
            "False Negative Ratio": metrics['false_negatives'][i] / (metrics['false_negatives'][i] + metrics['actual_ones'][i])
        }

    evaluation_metrics_by_threshold.append(metrics_for_threshold)

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
    "Date of Training": current_time,
    "Training File": file_paths[file_idx],
    "Data Range": {"Start": pd.Timestamp(dates[0]).strftime('%Y-%m-%d'), "End": pd.Timestamp(dates[-1]).strftime('%Y-%m-%d')},
    "Features Used": features,
    "Targets": targets,
    "Model Structure": str(model),
    "Model Parameters": {
        "Hidden Dimension": model.lstm.hidden_size,
        "Number of Layers": model.lstm.num_layers,
        "Dropout Rate": model.lstm.dropout,
    },
    "Loss Function": str(criterion),
    "Optimizer": {
        "Type": "Adam",
        "Learning Rate": 0.001,
        "Weight Decay": 1e-5
    },
    "Epochs": num_epochs,
    "Train Loss Over Epochs": train_losses,
    "Test Loss Over Epochs": test_losses,
    "Evaluation Metrics": evaluation_metrics_by_threshold,
}


with open(file_name, 'w') as file:
    json.dump(model_data, file, indent=4)

print(f"Saved model data to {file_name}")

# Given thresholds and evaluation_metrics_by_threshold from your code
thresholds = [metrics["Threshold"] for metrics in evaluation_metrics_by_threshold]

# Extracting the data for each target
fp_ratios = {target: [] for target in targets}
fn_ratios = {target: [] for target in targets}
fp_counts = {target: [] for target in targets}
fn_counts = {target: [] for target in targets}

for metrics in evaluation_metrics_by_threshold:
    for target in targets:
        fp_ratios[target].append(metrics[target]['False Positive Ratio'])
        fn_ratios[target].append(metrics[target]['False Negative Ratio'])
        fp_counts[target].append(metrics[target]['False Positives'])
        fn_counts[target].append(metrics[target]['False Negatives'])

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

for idx, target in enumerate(targets):
    axs[idx].plot(thresholds, fp_ratios[target], label=f'FP Ratio (FP: {fp_counts[target][idx]})')
    axs[idx].plot(thresholds, fn_ratios[target], label=f'FN Ratio (FN: {fn_counts[target][idx]})')
    
    axs[idx].set_title(f'Prediction over {target}')
    axs[idx].set_xlabel('Threshold')
    axs[idx].set_ylabel('Ratio')
    axs[idx].legend()

plt.tight_layout()
plt.show()


# from sklearn.metrics import confusion_matrix
# # Simulated data
# # y_true = np.random.randint(0, 2, 1000)
# # y_pred_probabilities = np.random.rand(1000)

# def compute_ratio(y_true, y_pred):
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     if fp == 0:
#         return np.inf  # Return infinity if false positive is zero
#     return fn / fp

# thresholds = np.linspace(0, 1, 100)
# # prediction_periods = [10, 30, 60]

# fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# for idx, hours in enumerate(targets):
#     ratios = []
#     abs_fns = []
#     abs_fps = []

#     # Extract data for the current prediction period
#     # current_y_true = y_true[:hours]
#     # current_y_pred_prob = y_pred_probabilities[:hours]

#     current_y_true = all_actual[idx]
#     current_y_pred_prob = all_predictions[idx]

#     for threshold in thresholds:
#         y_pred = (current_y_pred_prob > threshold).astype(int)
#         ratios.append(compute_ratio(current_y_true, y_pred))
#         _, fp, fn, _ = confusion_matrix(current_y_true, y_pred).ravel()
#         abs_fns.append(fn)
#         abs_fps.append(fp)

#     axs[idx].plot(thresholds, ratios, label=f'False Negative/False Positive Ratio')
#     axs[idx].set_title(f'Prediction Period: {hours} hours')
#     axs[idx].set_xlabel('Threshold')
#     axs[idx].set_ylabel('FN/FP Ratio')
#     axs[idx].legend([f'FN: {fn}, FP: {fp}' for fn, fp in zip(abs_fns, abs_fps)])

# plt.tight_layout()
# plt.show()

# 1. Plotting features over time
# fig, axes = plt.subplots(nrows=len(features), figsize=(14, 4*len(features)), sharex=True)

# for i, (ax, feature) in enumerate(zip(axes, features)):
#     ax.plot(dates, data[feature][seq_length:], label=feature)
#     if i == len(features) - 1:  # Only label the bottom-most x-axis
#         ax.set_xlabel('Date')
#     ax.set_ylabel('Value')
#     ax.set_title(f'Feature: {feature}')
#     ax.legend()
#     ax.grid(True)

# plt.tight_layout()
# plt.show()

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

# fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True, sharey=True)

# # Loop over prediction time frames (rows)
# for row, target in enumerate(targets):
#     # Loop over thresholds (columns)
#     for col, thresh in enumerate(thresholds):
#         ax = axes[row, col]
        
#         actual_values = [act[row] for act in all_actual[-len(y_test):]]
#         predicted_values = [pred[row] for pred in all_predictions[-len(y_test):]]
        
#         # Convert probability to binary labels based on threshold
#         predicted_labels = [(1 if p > thresh else 0) for p in predicted_values]
        
#         ax.plot(test_dates, actual_values, label="Actual", color="blue")
#         ax.plot(test_dates, predicted_labels, label="Predicted", color="red", alpha=0.7)
        
#         # If it's the first column, label the rows with the prediction timeframe
#         if col == 0:
#             ax.set_ylabel(f'{target} Target')
        
#         # If it's the last row, label the columns with the threshold
#         if row == 2:
#             ax.set_xlabel(f'Threshold: {thresh}')
        
#         ax.grid(True)

# # Tight layout to ensure no overlaps
# plt.tight_layout()
# plt.show()

fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True, sharey=True)

# Loop over prediction time frames (rows)
for row, target in enumerate(targets):
    # Loop over thresholds (columns)
    for col, thresh in enumerate(thresholds):
        ax = axes[row, col]
        
        actual_values = [act[row] for act in all_actual[-len(y_test):]]
        predicted_values = [pred[row] for pred in all_predictions[-len(y_test):]]
        
        # Convert probability to binary labels based on threshold
        predicted_labels = [(1 if p > thresh else 0) for p in predicted_values]
        
        # Compute errors for visualization
        errors = []
        for actual, predicted in zip(actual_values, predicted_labels):
            if actual == predicted:
                errors.append(0)
            elif actual == 0 and predicted == 1:
                errors.append(-1)
            else:
                errors.append(1)
        
        ax.plot(test_dates, errors, label="Error", color="green")
        
        # If it's the first column, label the rows with the prediction timeframe
        if col == 0:
            ax.set_ylabel(f'{target} Target')
        
        # If it's the last row, label the columns with the threshold
        if row == 2:
            ax.set_xlabel(f'Threshold: {thresh}')
        
        ax.axhline(0, color='black', linewidth=0.5) # Drawing the 0 line for clarity
        ax.set_ylim([-1.5, 1.5]) # Setting y-axis limits for clear visualization
        ax.grid(True)

# Tight layout to ensure no overlaps
plt.tight_layout()
current_time = datetime.now().strftime('%Y%m%d_%H%M')

plt.savefig(f'results/plots/errors_plot_{current_time}')
plt.show()

if not os.path.exists('results/plots'):
    os.makedirs('results/plots')
