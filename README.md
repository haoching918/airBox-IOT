# airBox-IOT

## About

A data visualization tool to monitor real time and history air data

## Interface overview

![](https://i.imgur.com/0ozRYkQ.png)


## Prereqest
Make sure to have python3 and node installed

## Load realtime data

### get device id
Run `get_device_id.py`, it will create a list of id indicate the current running airBox. It will also generate `info.json` to store update time

### load data of the device

Run `api.py` to call api to load 7 days data of each device, the data will be stored in `./data/device_week`

## data processing

### create discretize data

Run `discretize.py` to generate discetize 7 days data for each airBox to `./data/discretized_device_week`, also generate all airBox data at certain time in `./data/time_week`

### convert data to geojson

Run `datapreprocessing.js` to create geojson for leaflet to visualize

## Run

run the `index.html` with live server

