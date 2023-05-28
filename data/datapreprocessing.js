var interpolate = require('@turf/interpolate');
const fs = require('fs').promises;



function genDate(delimiter, dateStr, curTime) {
    var splitDateStr = dateStr.split(delimiter)
    return new Date(splitDateStr[0], splitDateStr[1] - 1, splitDateStr[2], curTime * 3 + 3)
}
Date.prototype.addDays = function (days) {
    const date = new Date(this.valueOf())
    date.setDate(date.getDate() + days)
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

async function generate_geojson(fileName) {
    const data = await fs.readFile("./time_week/" + fileName + ".json", 'utf8')
    console.log(fileName + ".json complete reading")
    var timeDevicesData = JSON.parse(data)

    var feat = { type: "FeatureCollection", features: [] }
    timeDevicesData.data.forEach(d => {
        if (d.gps_lon > 120 && d.gps_lat < 25 && d.gps_lat > 22 && d["pm2.5"] < 100) {
            feat.features.push({
                geometry: {
                    type: "Point",
                    coordinates: [d.gps_lon, d.gps_lat]
                },
                type: "Feature",
                properties: { solRad: d["pm2.5"] }
            })
        }
    })
    var options = { gridType: 'hex', property: 'solRad', units: 'kilometers', weight: 2 };
    var grid = interpolate(feat, 1.5, options);
    var str = JSON.stringify(grid, null, 0)

    await fs.writeFile("./geojson/" + fileName + ".json", str, 'utf8')
    console.log(fileName + ".json complete writing")
}
async function main() {
    const data = await fs.readFile("../info.json", 'utf-8')
    info = JSON.parse(data)

    var newDateStr = info["update_datetime"].slice(0, 10)
    var newTimeScale = parseInt(info["update_datetime"].slice(11, 13)) / 3 - 1
    var newDate = genDate("-", newDateStr, newTimeScale)
    var startDate = newDate.addDays(-7)
    var curDate = newDate
    var curTimeScale = newTimeScale

    for (let i = 55; i >= 0; i--) {
        fileName = curDate.yyyymmdd() + "-" + String(curTimeScale)
        generate_geojson(fileName)
        if (--curTimeScale < 0) {
            curTimeScale += 8
            curDate = curDate.addDays(-1)
        }
    }
}
main()



