import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import xmltodict
import json

API_KEY = 'CWB-32C8A51A-122D-4081-9FF5-EF2F76293AE9'
        
def store_wind(data):
    url = data['url']
    r = requests.get(url)
    r.encoding = 'utf-8'
    jsonData = json.dumps(xmltodict.parse(r.content))
    fileName = data['dataTime'][:13].replace(" ", "-")
    with open(f'./data/wind/{fileName}.json', 'w') as f:
        f.write(jsonData)
    
r = requests.get(f"https://opendata.cwb.gov.tw/historyapi/v1/getMetadata/O-A0001-001?Authorization={API_KEY}&format=JSON")
r.encoding = 'utf-8'
dictRequest = json.loads(r.text)
if dictRequest['dataset']['success'] != 'true':
    print("api call failed")
data = dictRequest['dataset']['resources']['resource']['data']['time']

with ThreadPoolExecutor(max_workers=10) as executor:
    for i in tqdm(executor.map(store_wind, data), total=len(data)): pass