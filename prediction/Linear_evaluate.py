import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
import os
import datetime as dt
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('./train_data/data.csv')
df = df.sort_values(by=['timestamp'])
df = df.drop(columns=['Unnamed: 0'])

# imputation
for column in df.columns[2:]:
  if np.any(df[column] <= 0):  # Check if column contains negative values
      column_mean = df[column].mean()  # Calculate column mean
      df[column] = np.where(df[column] <= 0, column_mean, df[column])  # Replace negative values with mean
      
# normalize
scaler = MinMaxScaler()
target_columns = ['temp','WDSD','pm2.5']  # Select only numeric columns
for column in target_columns:
    df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
    
class PM25Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
datasets = []
for device_id, group in df.groupby('deviceId'):

    group = group.drop(columns=['deviceId', 'timestamp'])  # Drop non-feature columns
    group_values = group.values # 24 * 5

    dataset = PM25Dataset(group_values)
    datasets.append(dataset)


dataset = ConcatDataset(datasets)
dataset, _ = torch.utils.data.random_split(dataset, [len(dataset), 0])
data_loader = DataLoader(dataset, batch_size=1333, shuffle=False)

# init device to store data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Linear Regression model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        out = self.fc(x)
        return out

file = './model/Linear_model.pt'
model = torch.load(file)
criterion = nn.MSELoss()

model.eval()  # Set the model to evaluation mode

week_loss = {
    '06-01' : [],
    '06-02' : [],
    '06-03' : [],
    '06-04' : [],
    '06-05' : [],
    '06-06' : [],
    '06-07' : [],
}
total_loss =[]
with torch.no_grad():  # Disable gradient tracking
    for i, inputs in enumerate(data_loader):

        inputs = inputs.float()
        inputs=inputs.to(device)
        outputs = model(inputs[:, :-1])
        targets = inputs[:, -1:]
        targets = targets.to(device)     
        loss = criterion(outputs, targets)
        
        week_loss['06-0'+str(1+i//24)].append(loss.item())
        total_loss.append(loss.item())
        

print(f'avg mse : {np.average(total_loss)}')
for i in week_loss:
    print(np.average(week_loss[i]))


