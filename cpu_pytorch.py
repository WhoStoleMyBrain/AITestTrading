# import torch 
import pandas as pd
import os
from glob import glob
import torch

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)

datapath = 'combined_data'
raw_datapath = 'raw_data'
# gemini_data_paths = glob(os.path.join(f'{datapath}', 'gemini_data_full*'))
gemini_raw_data_paths = glob(os.path.join(f'{raw_datapath}', 'Gemini*_1h.csv'))
# gemini_data_paths.sort()
# gemini_data = pd.read_csv(gemini_data_paths[-1])
# print(gemini_data.head(10))


crypto_name = gemini_raw_data_paths[0].split('_')[-2][:-3]
df = pd.read_csv(gemini_raw_data_paths[0], skiprows=1)
df = df.drop(columns=['date', 'symbol', f'Volume {crypto_name}'])
timeseries = df[["open", "high", "low", "close", "Volume USD"]].values.astype('float32')
# print(df.head(10))

train_size = int(len(timeseries)*0.67)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

lookback = 1
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


...
import torch.nn as nn

class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 5)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
    

import numpy as np
import torch.optim as optim
import torch.utils.data as data

model = AirModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

n_epochs = 2000
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))