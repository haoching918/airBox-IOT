import json
import datetime as dt

def getScale(hour):
    return str(hour // 3)
    if t == 0:
        return str(t)
    elif t == 1:
        return str(t)
    elif t == 2:
        return str(t)
    elif t == 3:
        return str(t)
    elif t == 4:
        return str(t)
    elif t == 5:
        return str(t)
    elif t == 6:
        return str(t)
    elif t == 7:
        return str(t)

def initDeviceData():
    # history per airbox
    deviceWeek = {
        'deviceId': None,
        'siteName': None,
        'gps_lon' : None,
        'gps_lat' : None,
        'startTime': startDatetime.strftime("%Y-%m-%d %H:%M:%S"),
        'endTime': endDatetime.strftime("%Y-%m-%d %H:%M:%S"),
        'data': {}
    }
    dateData = {}
    for i in range(8):
        scaleData = {}
        for j in range(8):
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

# set cur time         
with open('./info.json', 'r') as f:
    info = json.load(f)
endDatetime = dt.datetime.strptime(info['update_datetime'][:13], '%Y-%m-%d %H') # current date time without min and sec
startDatetime = endDatetime - dt.timedelta(days=7)

# get all device id
with open('./data/all_device_id.json','r') as f:
    devicesIdJson = json.load(f)
devicesId = devicesIdJson['id']

allDeviceWeek = [] # store discretized device data of a week in discretized_device_week

cnt = 1
for i in devicesId:
    with open(f"./data/device_week/{i}.json", 'r') as f:
        weekData = json.load(f)
    if weekData['num_of_records'] == 0:
        continue
    deviceWeek = initDeviceData()
    deviceWeek['deviceId'] = weekData['device_id']
    o =  list(weekData['feeds'][0]['AirBox'][0].values())[0]
    deviceWeek['siteName'] = o['SiteName']
    deviceWeek['gps_lat'] = o['gps_lat']
    deviceWeek['gps_lon'] = o['gps_lon']
    
    for d in weekData['feeds'][0]['AirBox']:
        data = list(d.values())[0]
        deviceDatetime = dt.datetime.strptime(data['timestamp'], '%Y-%m-%dT%H:%M:%SZ')
        if (deviceDatetime < startDatetime or deviceDatetime > endDatetime):
            continue
        if (data['s_d0'] > 100):
            continue
        deviceWeek['data'][str(deviceDatetime.date())][getScale(deviceDatetime.hour)]['pm2.5'].append(data['s_d0'])
        deviceWeek['data'][str(deviceDatetime.date())][getScale(deviceDatetime.hour)]['humidity'].append(data['s_h0'])
        deviceWeek['data'][str(deviceDatetime.date())][getScale(deviceDatetime.hour)]['temperature'].append(data['s_t0'])

    fillAvg(deviceWeek)
    allDeviceWeek.append(deviceWeek)
    
    jsonData = json.dumps(deviceWeek)
    with open(f'./data/discretized_device_week/{i}.json', 'w') as f:
        f.write(jsonData)
    print(f'{cnt} ./data/discretized_device_week/{i}.json saved')
    cnt += 1

# store all discretized device data of certain time scale in time_week
for i in range(8):
    for j in range(8):
        curDatetime = (startDatetime + dt.timedelta(days=i)).date()
        timeWeek = {
                'timeStamp': str(curDatetime),
                'timeScale' : j,
                'data' : []                
            }
        for d in allDeviceWeek:
            o = d['data'][str(curDatetime)][str(j)]
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
        with open(f'./data/time_week/{curDatetime}-{j}.json', 'w') as f:
            f.write(jsonData)
        print(f'{i*7+j} ./data/time_week/{curDatetime}-{j}.json saved')
            
# def grid(timeWeekData):
#     feat = { 'type' : 'FeatureCollection', 'features': [] }
#     for d in timeWeekData:
#         feat['features'].append({ 
#                                     'geometry': {
#                                             'type': 'Point',
#                                             'coordinates' : [d['gps_lon'], d['gps_lat']] 
#                                         }, 
#                                     'type': 'Feature', 
#                                     'properties': { 'solRad': d['pm2.5'] } 
#                                 })
#     options = {
#         'gridType' : 'hex',
#         'property' : 'solRad',
#         'units' : 'miles',
#         'weight' : 6
#     }
#     grid = interpolate(feat, 1, options)

# def interpolate():