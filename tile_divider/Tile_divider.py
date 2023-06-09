#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[1]:


import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import glob
import os



# ## Dividing DEM tiles


from shapely import geometry
from rasterio.mask import mask

# Takes a Rasterio dataset and splits it into squares of dimensions squareDim * squareDim
def splitImageIntoCells(img, filename, squareDim):
    numberOfCellsWide = img.shape[1] // squareDim
    numberOfCellsHigh = img.shape[0] // squareDim
    x, y = 0, 0
    count = 0
    for hc in range(numberOfCellsHigh):
        y = hc * squareDim
        for wc in range(numberOfCellsWide):
            x = wc * squareDim
            geom = getTileGeom(img.transform, x, y, squareDim)
            getCellFromGeom(img, geom, filename, count)
            count = count + 1

# Generate a bounding box from the pixel-wise coordinates using the original datasets transform property
def getTileGeom(transform, x, y, squareDim):
    corner1 = (x, y) * transform
    corner2 = (x + squareDim, y + squareDim) * transform
    return geometry.box(corner1[0], corner1[1],
                        corner2[0], corner2[1])

# Crop the dataset using the generated box and write it out as a GeoTIFF
def getCellFromGeom(img, geom, filename, count):
    crop, cropTransform = mask(img, [geom], crop=True)
    writeImageAsGeoTIFF(crop,
                        cropTransform,
                        img.meta,
                        img.crs,
                        filename+"_"+str(count))

# Write the passed in dataset as a GeoTIFF
def writeImageAsGeoTIFF(img, transform, metadata, crs, filename):
    metadata.update({"driver":"GTiff",
                     "height":img.shape[1],
                     "width":img.shape[2],
                     "transform": transform,
                     "crs": crs})
    with rasterio.open(filename+".tif", "w", **metadata) as dest:
        dest.write(img)


# In[35]:


dirpath = r'SwedenData/Raw_data/Dem_tiles/3DAlos_big/'
search_criteria = "*/*DSM.tif"
q = os.path.join(dirpath, search_criteria)
dem_tiles_paths = glob.glob(q) 


# In[36]:


print(len(dem_tiles_paths))


# In[37]:


print('Starting to create dem tiles of size 100')


# In[38]:


# for multiple dem tiles
import pathlib

for dem_path in dem_tiles_paths:
    file_name = dem_path.split('/')[-1].split('.')[0]
    
    #create directory
    directory = file_name
    parent_dir = "/home/azamat.kaibaldiyev/SwedenData/100tiles_all/dem/"
    new_folder_path = os.path.join(parent_dir, directory)
    pathlib.Path(new_folder_path).mkdir(exist_ok=True)
    print("Directory '%s' created" %directory)
    
    #divide and save in directory
    one_dem = rasterio.open(dem_path)
    splitImageIntoCells(one_dem, new_folder_path+'/'+file_name, 100)




# # Finding corresponding sat to dem tiles and saving it

# In[ ]:


new_repr_path = '/home/azamat.kaibaldiyev/SwedenData/Raw_data/Reprojected_MERGED_ALL_SAT/Reprojected_MERGED_ALL_SAT.tif'
reproj_sat = rasterio.open(new_repr_path)
reproj_sat.meta


# In[22]:


from rasterio.features import shapes
from rasterio.mask import mask
from rasterio.io import MemoryFile
from rasterio.merge import merge

def create_dataset(data, crs, transform):
    # Receives a 2D array, a transform and a crs to create a rasterio dataset
    memfile = MemoryFile()
    dataset = memfile.open(driver='GTiff', height=data.shape[0], width=data.shape[1], count=1, crs=crs, 
                           transform=transform, dtype=data.dtype)
    dataset.write(data, 1)
        
    return dataset



# In[28]:


from pathlib import Path
import pathlib
import numpy as np

dem_folders_paths = glob.glob('/home/azamat.kaibaldiyev/SwedenData/100tiles_all/dem/*')

for folder_dir in dem_folders_paths:
    
    # iterate over files in that directory
    images = Path(folder_dir).glob('*.tif')
    new_sat_folder_path = '/home/azamat.kaibaldiyev/SwedenData/100tiles_all/sat/' + folder_dir.split('/')[-1]
    pathlib.Path(new_sat_folder_path).mkdir(exist_ok=True)
    
    for image_path in images:
        try:
            end_name = str(image_path).split('/')[-1].split('.')[0]
            filename = new_sat_folder_path+'/' + end_name
            temp_dem_path = image_path
            temp_dem = rasterio.open(temp_dem_path)
            extents, _ = next(shapes(np.zeros_like(temp_dem.read()), transform=temp_dem.profile['transform']))
            cropped, crop_transf = mask(reproj_sat, [extents], crop=True)
            cropped_ds = create_dataset(cropped[0], reproj_sat.crs, crop_transf)
            writeImageAsGeoTIFF(cropped,
                                crop_transf,
                                cropped_ds.meta,
                                cropped_ds.crs,
                                filename)
            print('Sat image '+filename+' is saved')
        except:
            print('----Outside of zone: '+ end_name)
        




