#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import rasterio
import pandas as pd


df_labels =  pd.read_csv('Tasmania_ohara_labels/image_labels.data',skiprows = np.arange(15),sep = "\t")
                    #names = ['row_id','timestamp','left_image','right_image','cluster_id'], )
df_labels.rename(columns = {'left_image':'Left image name', 'right_image':'Right image name'}, inplace = True)
df_labels['Left image name'] = df_labels['Left image name'].str.strip()
df_labels['Right image name'] = df_labels['Right image name'].str.strip()


stereo_data_cols_names = ['Pose identifier', 'timestamp', 'Latitude', 'Longitude', 'X position (North)',
                         'Y position (East)', 'Z position (Depth)','X-axis Euler angle', 'Y-axis Euler angle',
                         'Z-axis Euler angle', 'Left image name', 'Right image name', 'Vehicle altitude', 
                         'Approx. bounding image radius', 'Likely trajectory cross-over point']


df_coords =  pd.read_csv('Tasmania_ohara_labels/stereo_pose_est.data',skiprows = np.arange(57),sep = "\t",
                    names = stereo_data_cols_names)
df_coords['Left image name'] = df_coords['Left image name'].str.strip()
df_coords['Right image name'] = df_coords['Right image name'].str.strip()


df_merged = pd.merge(df_labels, df_coords, on='Left image name')

from pyproj import Proj, transform

Pfrom = Proj(init='epsg:4326')
Pto = Proj(init='epsg:32755')
long_32755,lat_32755 = transform(Pfrom, Pto, df_merged['Longitude'], df_merged['Latitude'])

df_merged['long_32755'] = long_32755
df_merged['lat_32755'] = lat_32755
df_merged['cluster_id'] = df_merged['cluster_id'].replace([1, 2, 3 ,4,5, 6, 7, 8], [0, 1, 2, 3 ,4,5, 6, 7])


import rasterio
import numpy as np


# import rioxarray and shapley
import rioxarray as riox
from shapely.geometry import Polygon
 
# Read raster using rioxarray
raster = riox.open_rasterio('Tasmania_ohara_labels/fort1.tif')

filename_label = 'Tasmania_ohara_labels/cropped_dems_32_with_labels/'

for idx in range(len(df_merged['cluster_id'])):
    radius = 31 #in meters   #df_merged['Approx. bounding image radius'][idx]
    upper_left_x = df_merged['long_32755'][idx]- radius
    upper_left_y = df_merged['lat_32755'][idx]+ radius
    lower_right_x = df_merged['long_32755'][idx]+ radius
    lower_right_y = df_merged['lat_32755'][idx]- radius

    upper_right_x = df_merged['long_32755'][idx]+ radius
    upper_right_y = df_merged['lat_32755'][idx]+ radius
    lower_left_x = df_merged['long_32755'][idx]- radius
    lower_left_y = df_merged['lat_32755'][idx]- radius

    # Shapely Polygon  to clip raster
    geom = Polygon([[upper_left_x,upper_left_y], [upper_right_x,upper_right_y], [lower_right_x,lower_right_y], [lower_left_x,lower_left_y]])

    # Use shapely polygon in clip method of rioxarray object to clip raster
    clipped_raster = raster.rio.clip([geom],all_touched=True)

    # Save clipped raster
    label_idx = df_merged['cluster_id'][idx]
    clipped_raster.rio.to_raster(filename_label + 'img'+str(idx)+'_label'+str(label_idx)+'.tiff')

    cropped_raster = rasterio.open(filename_label + 'img'+str(idx)+'_label'+str(label_idx)+'.tiff')
    #print(cropped_raster.read().shape)
    if cropped_raster.read().shape!=(1, 32, 32):
        print(cropped_raster.read().shape)
    if idx%100==0:
        print('________________passed 100')
    #show(cropped_raster)







