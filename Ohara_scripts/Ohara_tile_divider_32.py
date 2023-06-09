#!/usr/bin/env python
# coding: utf-8

# In[1]:



import rasterio
from rasterio.windows import Window

filename = 'Tasmania_ohara_labels/cropped_dems_32/'

#for row in range(raster.shape[1]):
#    for col in range(raster.shape[2]):

raster = rasterio.open('Tasmania_ohara_labels/fort1.tif').read()
out_of_zone_val = raster[0,0,0]
print(raster.shape)

window_size = 32
step_size = 16


meta = {'driver': 'GTiff',
 'dtype': 'float32',
 'nodata': None,
 'width': window_size,
 'height': window_size,
 'count': 1,
 #'crs': CRS.from_epsg(32755),
 #'transform': Affine(2.0, 0.0, 579323.38,
       # 0.0, -2.0, 5229344.72)
       }


print('Starting 16')
for row in range(0, raster.shape[1], step_size):
    for col in range(0, raster.shape[2], step_size):
        
        col_idx = col
        row_idx = row
        
        with rasterio.open('Tasmania_ohara_labels/fort1.tif') as src:
             cropped_part = src.read(1, window=Window(col_idx, row_idx, window_size, window_size))
        
        
        if out_of_zone_val not in cropped_part:
            if cropped_part.shape == (window_size, window_size):
                out_fp = filename+'ohara_col'+str(col)+'_row'+str(row)
                with rasterio.open(out_fp, "w", **meta) as dest:
                    dest.write(cropped_part, indexes = 1)
                
print('Finished 16')




