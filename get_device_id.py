import json
import gzip
import requests
from datetime import datetime

def valid(d):
    if d['s_d0'] == -1:
        return False
    if d['gps_lon'] < 120 or d['gps_lon'] > 122:
        return False
    if d['gps_lat'] < 22:
        return False
    return True

# get gzip of last-all-airbox.josn
url = 'https://pm25.lass-net.org/data/last-all-airbox.json.gz'
r = requests.get(url, allow_redirects=True)
open('./data/last-all-airbox.json.gz', 'wb').write(r.content)

with gzip.open('./data/last-all-airbox.json.gz', 'rb') as f:
    data = json.load(f)

idDict = {
    'deviceNum': 0,
    'id': []
}

for d in data['feeds']:
    if valid(d):
        idDict['id'].append(d['device_id'])
        idDict['deviceNum'] += 1

jsonData = json.dumps(idDict)
with open('./data/all_device_id.json', mode='w') as f:
    f.write(jsonData)
    
timeScale = 3   # set time scale for discretize history data
info = {
    'update_datetime' : datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
    'time_scale' : timeScale
}
jsonData = json.dumps(info, indent=4)
with open('./info.json', 'w') as f:
    f.write(jsonData)