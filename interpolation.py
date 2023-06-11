import json
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import matplotlib.colors as colors
import cv2
import time
import os
import gc

# assign directory
directory = './data/time_week/'
 
def interpolate(path,filename):
    print('start',filename,'read'),
    with open(path + filename, 'r') as file:
        data = json.load(file)
        file.close()
    # decode json file data
    points = np.array([(d['gps_lon'], d['gps_lat'], d['pm2.5']) for d in data['data']])
    data = None
    grid_size = 0.005
    x_min,x_max = 118.21,122
    y_min,y_max = 21.895,26.5
    x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, grid_size),
                                np.arange(y_min, y_max, grid_size))
    grid_points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    # Calculate distances between GPS points and grid points
    distances = cdist(points[:, :2], grid_points).astype(np.float32)
    grid_points = None
    gc.collect()
    # Calculate weights using inverse distance weighting
    p = 2
    weights = 1/ distances ** p
    distances = None
    weights /= np.sum(weights,axis=0)
    # Interpolate values using weighted average
    interpolated_values =  np.dot(points[:,2].T,weights)

    # Create the image
    idw_image = interpolated_values.reshape(x_grid.shape)
    zero = np.zeros((idw_image.shape[0],int(1.5/grid_size)),dtype = float)
    idw_image = np.concatenate((idw_image[::-1],zero),axis= 1)
    zero = None
    interpolated_values = None
    weights = None
    # Define the PM2.5 AQI colors and corresponding values
    values_aqi = np.arange(500)

    # Create a smooth colormap that resembles the PM2.5 AQI color scale
    colors_smooth = [(0/255, 228/255, 0/255), 
                    (255/255, 255/255, 0/255), 
                    (255/255, 126/255, 0/255), 
                    (255/255, 0/255, 0/255), 
                    (153/255, 0/255, 76/255),
                    (153/255, 0/255, 76/255),
                    (126/255, 0/255, 35/255),
                    (126/255, 0/255, 35/255), 
                    (126/255, 0/255, 35/255), 
                    (126/255, 0/255, 35/255),  
                    (80/255, 0/255, 0/255)]
    cmap_aqi_smooth = colors.LinearSegmentedColormap.from_list('pm2.5_aqi_smooth', colors_smooth, N=500)
    norm_aqi = colors.BoundaryNorm(values_aqi, cmap_aqi_smooth.N)
    plt.imshow(idw_image, extent=[x_min, x_max, y_min, y_max], cmap=cmap_aqi_smooth, norm=norm_aqi)
    plt.axis('off')
    idw_image = None
    if not os.path.exists('./interpolate_image/'):
        os.mkdir('./interpolate_image/')
    img_name = './interpolate_image/' + filename[:len(filename) - 5] + '.png'
    plt.savefig(img_name, dpi=1200, bbox_inches='tight',pad_inches = 0)
    img = cv2.imread(img_name)
    img = cv2.resize(img,(2581,2464),interpolation= 1)
    cv2.imwrite(img_name,img)
    print('success',img_name,'write')
    return


if __name__ == '__main__':
    arr = []
    for filename in os.listdir('./interpolate_image/'):
        filename[:len(filename) -4 ] + 'json'
        arr.append(filename[:len(filename) -4 ] + '.json')
    print(arr)
    for filename in os.listdir(directory):
        if filename in arr:
            continue
        interpolate(directory,filename)
        gc.collect()