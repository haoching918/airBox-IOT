import requests
import json
from datetime import datetime

start = datetime.now()

f = open('./data/all_device_id.json')
data = json.load(f)
print(f"total devices : {data['deviceNum']}")
for i, id in enumerate(data['id']):
    r = requests.get(f'https://pm25.lass-net.org/API-1.0.0/device/{id}/history/?format=JSON')
    r.encoding = 'big5'
    jsondata = json.dumps(r.json())
    with open(f"./data/device_week/{id}.json", "w") as outfile:
        outfile.write(jsondata)
    print(f'{i} ./data/device_week/{id}.json saved')
  
end = datetime.now()  
print(f'start time : {start}\nend time : {end}')