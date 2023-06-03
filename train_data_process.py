import pandas as pd
import numpy as np
import json
import os
from scipy.spatial.distance import cdist

def load_json(filename):
    with open(filename,'r') as f:
        FILE = json.load(f)
    return FILE


filepath_src = './data/time_week'
filepath_src_2 = './data/wind'
filepath = './data/traindata'

if not os.path.exists(filepath):
    os.mkdir(filepath)

time_week_file = []
wind_file = []
for filename in os.listdir(filepath_src):
    time_week_file.append(filename)
    
for filename in os.listdir(filepath_src_2):
    wind_file.append(filename)

time_week_file.sort()
wind_file.sort()
df = pd.DataFrame(columns=['deviceId','timestamp','lat','lon','temp','pm2.5','WDSD'])
for index,filenames in enumerate(zip(time_week_file,wind_file)):
    filename_time,filename_wind = filenames
    time_week_json = load_json(filepath_src + '/' + filename_time)
    try:
        f = time_week_file[index+1]
    except:
        break
    time_week_json_next = load_json(filepath_src + '/' + time_week_file[index+1])
    wind_json = load_json(filepath_src_2 + '/' + filename_wind)
    next_pm25 = np.array([(device['pm2.5']) for device in time_week_json_next['data']])
    device_pos = np.array([(device['gps_lat'],device['gps_lon'],device['temperature'],device['pm2.5']) for device in time_week_json['data']])
    id_list = np.array([(device['deviceId'],filename_time) for device in time_week_json['data']])
    wind_pos = np.array([(float(wind_device['lat']),float(wind_device['lon']),float(wind_device['weatherElement'][3]['elementValue']['value'])) for wind_device in wind_json['cwbopendata']['location']])
    distance = cdist(device_pos[:,:2],wind_pos[:,:2])
    wind_list = np.array([(wind_pos[d2d.argmin()][2]) for d2d in distance])
    wind_list = wind_list.reshape(-1,1)
    next_pm25 = next_pm25.reshape(-1,1)
    combine = np.hstack([id_list,device_pos,wind_list,next_pm25])
    df = pd.concat([df,pd.DataFrame(combine,columns=['deviceId','timestamp','lat','lon','temp','pm2.5','WDSD','next_pm2.5'])],axis=0)

df.to_csv(filepath + '/testtrainfile.csv')
