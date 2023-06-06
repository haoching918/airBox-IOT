var interpolate = require('@turf/interpolate');
const fs = require('fs').promises;



// async function generate(i, j) {
//     const data = await fs.readFile(`/mnt/d/111-1/airBoxIot/project/data/time_week/2022-12-${i > 9 ? i : `0${i}`}-${j}.json`, 'utf8')
//     console.log("time_week/2022-12-${i > 9 ? i : `0${i}`}-${j}.json complete reading")
//     var timeDevicesData = JSON.parse(data)


//     var feat = { type: "FeatureCollection", features: [] }
//     timeDevicesData.data.forEach(d => {
//         if (d.gps_lon > 120 && d.gps_lat < 25 && d.gps_lat > 22 && d["pm2.5"] < 100) {
//             feat.features.push({
//                 geometry: {
//                     type: "Point",
//                     coordinates: [d.gps_lon, d.gps_lat]
//                 },
//                 type: "Feature",
//                 properties: { solRad: d["pm2.5"] }
//             })
//         }
//     })
//     var options = { gridType: 'hex', property: 'solRad', units: 'miles', weight: 6 };
//     var grid = interpolate(feat, 1, options);
//     var str = JSON.stringify(grid, null, 0)

//     await fs.writeFile(`/mnt/d/111-1/airBoxIot/project/data/geojson/2022-12-${i > 9 ? i : `0${i}`}-${j}.json`, str, 'utf8')
//     console.log("geojson/2022-12-${i > 9 ? i : `0${i}`}-${j}.json complete writing")
// }

// for (i = 0; i < 7; i++) {
//     for (j = 0; j < 8; j++) { generate(i, j) }
// }


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
        if (d.gps_lon < 122.015 && d.gps_lon > 118.2 && d.gps_lat < 26.4 && d.gps_lat > 21.8 && d["pm2.5"] < 100) {
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

    dateTimeStr = info["update_datetime"]
    timeScale = info["time_scale"]
    var newDateStr = dateTimeStr.slice(0, 10)
    var newTime = ~~(parseInt(dateTimeStr.slice(11, 13)) / timeScale) - 1
    var newDate = genDate("-", newDateStr, newTime)
    var startDate = newDate.addDays(-7)
    var curDate = newDate
    var curTime = newTime
    var totalTime = 24 / timeScale

    for (let i = totalTime * 7; i > 0; i--) {
        fileName = curDate.yyyymmdd() + "-" + (curTime >= 10 ? String(curTime) : "0" + String(curTime))
        generate_geojson(fileName)
        if (--curTime < 0) {
            curTime += totalTime
            curDate = curDate.addDays(-1)
        }
    }
}
main()