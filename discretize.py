import json
from tqdm import tqdm, trange
from concurrent.futures import ThreadPoolExecutor
import datetime as dt

def getScale(hour):
    return str(hour // timeScale)

def initDeviceData():
    # history per airbox
    deviceWeek = {
        'deviceId': None,
        'siteName': None,
        'gps_lon' : None,
        'gps_lat' : None,
        'startTime': startDatetime.strftime("%Y-%m-%d %H:%M:%S"),
        'endTime': endDatetime.strftime("%Y-%m-%d %H:%M:%S"),
        'timeScale': timeScale,
        'data': {}
    }
    dateData = {}
    for i in range(8):
        scaleData = {}
        for j in range(24//timeScale):
            scaleData[str(j)] = {'pm2.5' : [], 
                                 'avgPm2.5': None,
                                 'humidity': [],
                                 'avgHumidity': None,
                                 'temperature': [],
                                 'avgTemperature': None
                            }
        dateData[str(startDatetime.date() + dt.timedelta(days=i))] = scaleData
    deviceWeek['data'] = dateData
    
    return deviceWeek
    
def fillAvg(deviceWeek):
    for day in deviceWeek['data'].values():
        for time in day.values():
            total = 0
            for i in time['pm2.5']:
                total += i
            time['avgPm2.5'] = total / len(time['pm2.5']) if len(time['pm2.5']) != 0 else 0 # check divide zero
            total = 0
            for i in time['humidity']:
                total += i
            time['avgHumidity'] = total / len(time['humidity']) if len(time['humidity']) != 0 else 0
            total = 0
            for i in time['temperature']:
                total += i
            time['avgTemperature'] = total / len(time['temperature']) if len(time['temperature']) != 0 else 0
    return deviceWeek

# store discretized data of a given device in discretized_device_week
def store_device_week(id):
    with open(f"./data/device_week/{id}.json", 'r') as f:
        weekData = json.load(f)
    if weekData['num_of_records'] == 0:
        return
    deviceWeek = initDeviceData()
    deviceWeek['deviceId'] = weekData['device_id']
    o =  list(weekData['feeds'][0]['AirBox'][0].values())[0]
    deviceWeek['siteName'] = o['SiteName']
    deviceWeek['gps_lat'] = o['gps_lat']
    deviceWeek['gps_lon'] = o['gps_lon']
    
    for d in weekData['feeds'][0]['AirBox']:
        data = list(d.values())[0]
        deviceDatetime = dt.datetime.strptime(data['timestamp'], '%Y-%m-%dT%H:%M:%SZ')
        if deviceDatetime < startDatetime or deviceDatetime > endDatetime:
            continue
        if data['s_d0'] < 100 or data['s_d0'] > 0:
            deviceWeek['data'][str(deviceDatetime.date())][getScale(deviceDatetime.hour)]['pm2.5'].append(data['s_d0'])
        if data['s_t0'] > 0 or data['s_t0'] < 40:
            deviceWeek['data'][str(deviceDatetime.date())][getScale(deviceDatetime.hour)]['temperature'].append(data['s_t0'])
        deviceWeek['data'][str(deviceDatetime.date())][getScale(deviceDatetime.hour)]['humidity'].append(data['s_h0'])
        
    fillAvg(deviceWeek)   
    jsonData = json.dumps(deviceWeek)
    with open(f'./data/discretized_device_week/{id}.json', 'w') as f:
        f.write(jsonData)
    return deviceWeek

# store all discretized device data of certain day and time in time_week
def store_date(day, time):
    curDatetime = (startDatetime + dt.timedelta(days=day)).date()
    timeWeek = {
            'timeStamp': str(curDatetime),
            'timeScale' : time,
            'data' : []                
        }
    for d in allDeviceWeek:
        o = d['data'][str(curDatetime)][str(time)]
        data = {
            'deviceId' : d['deviceId'],
            'siteName' : d['siteName'],
            'gps_lat' : d['gps_lat'],
            'gps_lon' : d['gps_lon'],
            'pm2.5' : o['avgPm2.5'],
            'humidity' : o['avgHumidity'],
            'temperature' : o['avgTemperature']
        }
        timeWeek['data'].append(data)
    
    jsonData = json.dumps(timeWeek)
    if time >= 10:
        with open(f'./data/time_week/{curDatetime}-{time}.json', 'w') as f:
            f.write(jsonData)
    else :
        with open(f'./data/time_week/{curDatetime}-0{time}.json', 'w') as f:
            f.write(jsonData)
         
# set cur time         
with open('./info.json', 'r') as f:
    info = json.load(f)
endDatetime = dt.datetime.strptime(info['update_datetime'][:13], '%Y-%m-%d %H') # current date time without min and sec
startDatetime = endDatetime - dt.timedelta(days=7)
# get time scale
timeScale = info['time_scale']

# get all device id
with open('./data/all_device_id.json','r') as f:
    devicesIdJson = json.load(f)
devicesId = devicesIdJson['id']

allDeviceWeek = [] # store discretized device data of a week in discretized_device_week

print("storing discretized device week data")
with ThreadPoolExecutor(max_workers=10) as executor:
    for r in tqdm(executor.map(store_device_week, devicesId), total=len(devicesId)):
        allDeviceWeek.append(r)

print("storing time week data")
for day in trange(8):
    with ThreadPoolExecutor(max_workers=10) as executor:
        times = range(24//timeScale)
        days = [day for _ in times]
        executor.map(store_date, days, times)