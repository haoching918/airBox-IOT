import requests
import json
import os
from datetime import datetime
path_data = './data'
path_week = '/device_week'


start = datetime.now()

f = open(path_data + '/all_device_id.json')
data = json.load(f)
print(f"total devices : {data['deviceNum']}")
if not os.path.exists(path_data + path_week):
    os.mkdir(path_data + path_week)

for i, id in enumerate(data['id']):
    r = requests.get(f'https://pm25.lass-net.org/API-1.0.0/device/{id}/history/?format=JSON')
    r.encoding = 'big5'
    jsondata = json.dumps(r.json())
    with open(path_data + path_week + f"/{id}.json", "w") as outfile:
        outfile.write(jsondata)
    print(f'{i} '+ path_data + path_week + f'/{id}.json saved')
  
end = datetime.now()  
print(f'start time : {start}\nend time : {end}')