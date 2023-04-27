import requests
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

def make_request(id):
    r = requests.get(f'https://pm25.lass-net.org/API-1.0.0/device/{id}/history/?format=JSON')
    r.encoding = 'big5'
    jsondata = json.dumps(r.json())
    with open(f"./data/device_week/{id}.json", "w") as outfile:
        outfile.write(jsondata)

print("fetching data from pm25.lass-net.org")
start = datetime.now()
with open('./data/all_device_id.json') as f:
    data = json.load(f)
    print(f"total devices : {data['deviceNum']}")
    with ThreadPoolExecutor(max_workers=20) as executor:
        for i in tqdm(executor.map(make_request, data['id']), total= data['deviceNum']):
            pass

end = datetime.now()  
print(f'total time : {end-start}')