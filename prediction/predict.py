import pandas as pd
import numpy as np
import torch
import json
import os
import datetime as dt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

def load_json(filename):
    with open(filename,'r') as f:
        FILE = json.load(f)
    return FILE


filepath_src = '../data/time_week'
filepath_src_2 = '../data/wind'
filepath = './predict_result'

if not os.path.exists(filepath):
    os.mkdir(filepath)

info = load_json('../info.json')
airDatetime = dt.datetime.strptime(info['update_datetime'][:13], '%Y-%m-%d %H')
windDatetime = airDatetime + dt.timedelta(hours=9)
curDatetime = airDatetime

time_week_file = []
wind_file = []
sequence_length = 4
for i in range(sequence_length):
    airDatetime -= dt.timedelta(hours=1)
    air_file_name = airDatetime.strftime('%Y-%m-%d-%H')
    time_week_file.append(air_file_name)
    windDatetime -= dt.timedelta(hours=1)
    wind_file_name = windDatetime.strftime('%Y-%m-%d-%H')
    wind_file.append(wind_file_name)

time_week_file.sort()
wind_file.sort() 

df = pd.DataFrame(columns=['deviceId','timestamp','lat','lon','temp','pm2.5','WDSD'])
for index,filenames in enumerate(zip(time_week_file,wind_file)):
    filename_time,filename_wind = filenames
    time_week_json = load_json(filepath_src + '/' + filename_time + '.json')
    wind_json = load_json(filepath_src_2 + '/' + filename_wind + '.json')

    device_pos = np.array([(device['gps_lat'],device['gps_lon'],device['temperature'],device['pm2.5']) for device in time_week_json['data']])
    id_list = np.array([(device['deviceId'],filename_time) for device in time_week_json['data']])
    wind_pos = np.array([(float(wind_device['lat']),float(wind_device['lon']),float(wind_device['weatherElement'][3]['elementValue']['value'])) for wind_device in wind_json['cwbopendata']['location']])
    distance = cdist(device_pos[:,:2],wind_pos[:,:2])
    wind_list = np.array([(wind_pos[d2d.argmin()][2]) for d2d in distance])
    wind_list = wind_list.reshape(-1,1)
    combine = np.hstack([id_list,device_pos,wind_list])
    df = pd.concat([df,pd.DataFrame(combine,columns=['deviceId','timestamp','lat','lon','temp','pm2.5','WDSD'])],axis=0)

# imputation
for column in df.columns[2:]:
    df[column] = df[column].astype('float32')
    if np.any(df[column] <= 0):  # Check if column contains negative values
        column_mean = df[column].mean()  # Calculate column mean
        df[column] = np.where(df[column] <= 0, column_mean, df[column])  # Replace negative values with mean

# normalize
scaler = MinMaxScaler()
target_columns = ['temp','WDSD','pm2.5']  # Select only numeric columns
for column in target_columns:
    df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
  
class PM25Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(PM25Predictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

datasets = []
for device_id, group in df.groupby('deviceId'):
    group = group.drop(columns=['deviceId', 'timestamp'])
    group_values = group.values # 24 * 5
    datasets.append(group_values)

datasets = np.array(datasets)

datasets = torch.FloatTensor(datasets)

file = './model/LSTM_model.pt'
model = torch.load(file)

model.eval()  # Set the model to evaluation mode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # init device to store data
model = model.to(device)
datasets = datasets.to(device)

outputs = model(datasets)

fileName = curDatetime.strftime('%Y-%m-%d-%H')
predict_time_week = {
    'timeStamp': fileName,
    'data': [],
}

for i, (device_id, group) in enumerate(df.groupby('deviceId')):
    data = {
        'deviceId' : device_id,
        'gps_lat' : float(group['lat'].values[0]),
        'gps_lon' : float(group['lon'].values[0]),
        'pm2.5' : float(outputs[i]),
    }
    predict_time_week['data'].append(data)

jsonData = json.dumps(predict_time_week)
with open(f'{filepath}/{fileName}.json', 'w') as f:
    f.write(jsonData)