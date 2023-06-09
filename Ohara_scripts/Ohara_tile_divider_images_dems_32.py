
import glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import rasterio
import pandas as pd




csv_paths = sorted(glob.glob('Tasmania_ohara_labels/All_images_csv/*'))


df_csvs = pd.read_csv(csv_paths[0])

for idx, path in enumerate(csv_paths):
    if idx!=0:
        df_new = pd.read_csv(path)
        df_csvs = pd.concat([df_csvs, df_new])
        
df_csvs.rename(columns={"leftimage": "Left image name"}, inplace = True)
df_csvs['Left image name'] = df_csvs['Left image name'].str.strip().str.split('.').str[0]+'.tif'
df_csvs.reset_index(inplace = True)



import glob
imgs = sorted(glob.glob('Tasmania_ohara_labels/Ohara_images_from_prof_2/*/*'))
print(len(imgs))
df_images = pd.DataFrame(imgs, columns =['Left image name'])
df_images['Left image name'] = df_images['Left image name'].str.split('/').str[-1]



df_merged_csv_images = pd.merge(df_csvs, df_images, on = 'Left image name')




from pyproj import Proj, transform

Pfrom = Proj(init='epsg:4326')
Pto = Proj(init='epsg:32755')

long_32755,lat_32755 = transform(Pfrom, Pto, df_merged_csv_images['longitude'], df_merged_csv_images['latitude'])
df_merged_csv_images['long_32755'] = long_32755
df_merged_csv_images['lat_32755'] = lat_32755



df_merged = df_merged_csv_images
# import rioxarray and shapley
import rioxarray as riox
from shapely.geometry import Polygon
 
# Read raster using rioxarray
raster = riox.open_rasterio('Tasmania_ohara_labels/fort1.tif')

filename_label = 'Tasmania_ohara_labels/cropped_dems_32_imgs_100/'

for idx in range(len(df_merged)):
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
    #label_idx = df_merged['cluster_id'][idx]
    clipped_raster.rio.to_raster(filename_label + df_merged['Left image name'][idx])

#     cropped_raster = rasterio.open(filename_label + df_merged['Left image name'][idx])
#     print(cropped_raster.read().shape)
#     if cropped_raster.read().shape!=(1, 32, 32):
#         print(cropped_raster.read().shape)
    if idx%100==0:
        print('________________passed 100')
    #show(cropped_raster)