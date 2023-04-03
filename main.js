class Map {
	zoom = 8
	latlng = L.latLng(23.5, 121);
	constructor(data, grid) {
		this.timeDevicesData = data;
		// create taiwan map layer
		this.OpenStreetMap_Mapnik = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
			maxZoom: 19,
			attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
		});
		this.Stamen_Terrain = L.tileLayer('https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}{r}.{ext}', {
			attribution: 'Map tiles by <a href="http://stamen.com">Stamen Design</a>, <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
			subdomains: 'abcd',
			minZoom: 0,
			maxZoom: 18,
			ext: 'png'
		});

		this.airBoxPoints = []
		this.timeDevicesData.data.forEach(d => {
			if (d.gps_lon > 120 && d.gps_lat < 25 && d.gps_lat > 22 && d["pm2.5"] < 100) {
				this.airBoxPoints.push(L.marker([d.gps_lat, d.gps_lon]).bindPopup(d.siteName).on('click', function () {
					getDevice(d.deviceId)
				})) // create marker layer for device
			}
		})
		this.airBox = L.layerGroup(this.airBoxPoints)
		// create idw layer
		this.grid = grid;
		this.idw = L.geoJson(this.grid, {
			style: function (feature) {
				return {
					"color": "black",
					"fillColor": getColor(feature.properties.solRad),
					"opacity": 0,
					"fillOpacity": 0.5
				}
			}
		})

		this.draw()
	}
	draw() {
		this.maps = L.map('map', { layers: [this.Stamen_Terrain, this.idw] }).setView(this.latlng, this.zoom);
		this.drawLegend()
		this.baseMaps = {
			"街道圖": this.OpenStreetMap_Mapnik,
			"地形圖" : this.Stamen_Terrain,
		};
		this.overlayMaps = {
			"內插圖": this.idw,
			"空氣盒子站點": this.airBox,
		};

		this.layerControl = L.control.layers(this.baseMaps, this.overlayMaps, { position: 'bottomright' }).addTo(this.maps);
	}
	updateIdw(data) {
		this.zoom = this.maps.getZoom()
		this.latlng = this.maps.getCenter()
		this.grid = data
		this.idw = L.geoJson(this.grid, {
			style: function (feature) {
				return {
					"color": "black",
					"fillColor": getColor(feature.properties.solRad),
					"opacity": 0,
					"fillOpacity": 0.5
				}
			}
		})
		this.layerControl.remove()
		this.maps.remove()
		this.draw()
	}
	drawLegend() {
		var legend = L.control({ position: "bottomleft" });
		var div = L.DomUtil.create('div', 'info legend')
		var grades = [0, 11, 23, 35, 41, 47, 53, 58, 64, 70];
		legend.onAdd = function (map) {
			div.innerHTML += "<h4>PM2.5指標對照表</h4>";
			for (var i = 0; i < grades.length; i++) {
				div.innerHTML += "<i style= background:" + getColor(grades[i] + 1) + "></i> " + grades[i] + (grades[i + 1] ? "&ndash;" + grades[i + 1] + "<br>" : "+");
			}
			return div;
		}
		legend.addTo(this.maps);
	}
}
async function animate() {
	var date = startDate
	var time = newTimeScale + 1
	for (let i = 0; i < 56; i++) {
		setCurDate(date.yyyymmdd(), time)
		let path = "./data/geojson/" + date.yyyymmdd() + "-" + time + ".json"
		let response = await fetch(path)
		let newGrid = await response.json()
		myMap.updateIdw(newGrid)
		if (++time == 8)
			date = date.addDays(1)
		time %= 8
	}
}
async function updateCurDateIdw() {
	var date = document.getElementById("dateSelector").value
	var time = document.getElementById("timeSelector").value
	if (!validTime(date, time)) {
		alert("Invalid date")
		return
	}
	var newGrid = await fetch_json("./data/geojson/" + date + "-" + time + ".json")
	myMap.updateIdw(newGrid)
}
async function getDevice(deviceId) {
	var response = await fetch(`./data/discretized_device_week/${deviceId}.json`)
	var deviceWeekData = await response.json()
	myChart.setData(deviceWeekData)
}
function getColor(x) {
	return x < 11 ? '#9CFF9C' :
		x < 23 ? '#31FF00' :
			x < 35 ? '#31CF00' :
				x < 41 ? '#FFFF00' :
					x < 47 ? '#FFCF00' :
						x < 53 ? '#FF9A00' :
							x < 58 ? '#FF6464' :
								x < 64 ? '#FF0000' :
									x < 70 ? '#990000' :
										'#CE30FF'
}

class Linechart {
	deviceWeekData
	chart

	constructor(deviceWeekData) {
		this.deviceWeekData = deviceWeekData
		let chartData = {
			labels: [],
			datasets: [
				{
					label: "",
					data: [],
					fill: false,
					borderColor: "rgb(75, 192, 192)",
				},
				{
					label: "",
					data: [],
					fill: false,
					borderColor: "rgb(200, 0, 0)",
				},
				// {
				// 	label: "",
				// 	data: [],
				// 	fill: false,
				// 	borderColor: "rgb(0, 0, 200)",
				// },
			],
		}
		let config = {
			type: 'line',
			data: chartData,
			options: {
				responsive: true,
				layout: {
					padding: {
						top: 15,
					}
				},
				plugins: {
					legend: {
						position: 'top',
					},
					title: {
						display: true,
						text: "",
						color: 'black',
						font: {
							family: 'Times',
							size: 20,
							style: 'normal',
							lineHeight: 0
						}
					}
				},
				scales: {
					x: {
						display: true,
						title: {
							display: true,
							text: '時間',
							color: 'black',
							font: {
								family: 'Times',
								size: 20,
								style: 'normal',
								lineHeight: 1.2
							},
							padding: { top: 20, left: 0, right: 0, bottom: 0 }
						}
					},
					y: {
						display: true,
						title: {
							display: true,
							text: '數值',
							color: 'black',
							font: {
								family: 'Times',
								size: 20,
								style: 'normal',
								lineHeight: 1.2,
							},
							padding: { top: 20, left: 0, right: 0, bottom: 0 }
						}
					}
				}
			},
		};
		this.chart = new Chart(document.getElementById("myChart"), config)
		this.setTimeScale()
	}
	setTimeScale() {
		let labels = []
		let pmData = []
		let tempData = []
		let humidData = []
		for (let date = startDate; date.getTime() < newDate.getTime(); date = date.addHours(3)) {
			let dateStr = date.yyyymmdd()
			let timeStr = String(date.getHours() / 3)
			//labels.push(timeStr == "0" ? dateStr.slice(5) : timeStr)
			labels.push(dateStr.slice(5) + " " + timeStr)
			pmData.push(this.deviceWeekData.data[dateStr][timeStr]["avgPm2.5"])
			tempData.push(this.deviceWeekData.data[dateStr][timeStr].avgTemperature)
			// humidData.push(this.deviceWeekData.data[dateStr][timeStr].avgHumidity)
		}
		this.chart.options.plugins.title.text = this.deviceWeekData.siteName
		let chartData = this.chart.data
		chartData.labels = labels
		chartData.datasets[0].data = pmData
		chartData.datasets[0].label = "7 days PM2.5 （μg/m3）"
		chartData.datasets[1].data = tempData
		chartData.datasets[1].label = "7 days temperature ℃"
		// chartData.datasets[2].data = humidData
		// chartData.datasets[2].label = this.timeLabels[timeScale] + " humidity （RH）"

		this.chart.update()
	}
	setData(deviceWeekData) {
		this.deviceWeekData = deviceWeekData
		this.setTimeScale()
	}
	maxDate(date1, date2) {
		return date1.getTime() > date2.getTime() ? date1 : date2
	}

}
// function updateChartTime1() {
// 	myChart.setTimeScale(1)
// }
// function updateChartTime3() {
// 	myChart.setTimeScale(3)
// }
// function updateChartTime7() {
// 	myChart.setTimeScale(7)
// }

function genDate(delimiter, dateStr, curTime) {
	var splitDateStr = dateStr.split(delimiter)
	return new Date(splitDateStr[0], splitDateStr[1] - 1, splitDateStr[2], curTime * 3 + 3)
}
function validTime(dateStr, time) {
	var date = genDate("-", dateStr, time)
	if (date.getTime() <= startDate.getTime() || date.getTime() > newDate.getTime()) return false

	return true
}
// set cur date at html
function setCurDate(dateStr, time) {
	var dateSelector = document.getElementById("dateSelector")
	dateSelector.value = dateStr

	var timeSelector = document.getElementById("timeSelector")
	timeSelector.value = String(time)
}
function initSelector() {
	var dateSelector = document.getElementById("dateSelector")
	dateSelector.max = newDate.yyyymmdd()
	dateSelector.min = newDate.addDays(-7).yyyymmdd()
	document.getElementById("last_update").innerHTML = `最後更新時間 ${newDate.yyyymmdd()} ${(newTimeScale + 1) * 3}:00 GMT+0`
}

async function fetch_json(path) {
	var response = await fetch(path)
	var r = await response.json()
	return r
}
Date.prototype.addDays = function (days) {
	const date = new Date(this.valueOf())
	date.setDate(date.getDate() + days)
	return date
}
Date.prototype.addHours = function (hours) {
	const date = new Date(this.valueOf())
	date.setHours(date.getHours() + hours)
	return date
}
Date.prototype.yyyymmdd = function () {
	var mm = this.getMonth() + 1; // getMonth() is zero-based
	var dd = this.getDate();

	return [this.getFullYear(),
	(mm > 9 ? '' : '0') + mm,
	(dd > 9 ? '' : '0') + dd
	].join('-');
};

var info = await fetch_json("./info.json")
var newDateStr = info["update_datetime"].slice(0, 10) // yyyy-mm-dd
var newTimeScale = parseInt(info["update_datetime"].slice(11, 13)) / 3 - 1 // 0-7 int
var newDate = genDate("-", newDateStr, newTimeScale)
var startDate = newDate.addDays(-7)

setCurDate(newDateStr, newTimeScale)
initSelector()

var timeDevicesData = await fetch_json("./data/time_week/" + newDateStr + "-" + String(newTimeScale) + ".json")
var deviceWeekData = await fetch_json("./data/discretized_device_week/74DA38F70318.json")
var geojson = await fetch_json("./data/geojson/" + newDateStr + "-" + String(newTimeScale) + ".json")

var myMap = new Map(timeDevicesData, geojson)
var myChart = new Linechart(deviceWeekData)
myChart.chart.canvas.onclick = function (evt) {
	var activePoint = myChart.chart.getElementsAtEventForMode(evt, 'nearest', { intersect: true }, false)
	var dataIndex = activePoint[0].index
	var dateStr = myChart.chart.data.labels[dataIndex] // mm-dd t
	setCurDate(`${newDateStr.slice(0, 4)}-${dateStr.slice(0, 5)}`, parseInt(dateStr.slice(6)))
	updateCurDateIdw()
}

document.getElementById('submitDate').onclick = updateCurDateIdw
document.getElementById('playAnimation').onclick = animate
// document.getElementById('oneDay').onclick = updateChartTime1
// document.getElementById('threeDays').onclick = updateChartTime3
// document.getElementById('sevenDays').onclick = updateChartTime7


