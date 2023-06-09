#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Setup

# In[1]:


import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm.notebook import tqdm


# In[2]:


device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print(device)



with open('remove_list_indexes.npy', 'rb') as f:
    remove_list = np.load(f)



with open('remove_water_list.npy', 'rb') as f:
    remove_water_list = np.load(f)


from torch.utils.data import Dataset, DataLoader
import glob
import rasterio
class SwedishDataset(Dataset):
    def __init__(self,  transform  = None, remove_list = None,remove_water_list = None):
        self.imgs_path = "SwedenData/100tiles_all/"
        self.file_list_dem1 = sorted(glob.glob(self.imgs_path + 'dem/'+ "*/*"))
        self.file_list_sat1 = sorted(glob.glob(self.imgs_path + 'sat/'+ "*/*"))
        
        
        search_list = ['/'.join(ele.split('/')[-2:]) for ele in self.file_list_dem1]
        test_list = ['/'.join(ele.split('/')[-2:]) for ele in self.file_list_sat1]
        self.file_list_dem1 = [ele for ele in search_list if ele in test_list]

        self.data = []
        for img_path_dem,img_path_sat in zip(self.file_list_dem1,self.file_list_sat1):
          self.data.append(["SwedenData/100tiles_all/dem/"+img_path_dem,img_path_sat])
          #self.data.append([img_path_dem,img_path_sat])
        
        

        self.transform = transform

        if remove_list is not None:
            self.data = np.array(self.data)
            mask = np.ones(len(self.data), dtype=bool)
            mask[remove_list] = False
            self.data = self.data[mask]
            
        if remove_water_list is not None:
            mask = np.ones(len(self.data), dtype=bool)
            mask[remove_water_list] = False
            self.data = self.data[mask]

        
    def __len__(self):
        return len(self.data)


          
    def __getitem__(self, idx):
        img_path_dem, img_path_sat = self.data[idx]
        img_raster_dem = rasterio.open(img_path_dem).read()#[:,:96,:96]
        img_raster_sat = rasterio.open(img_path_sat).read()
        
        return [img_raster_dem ,img_raster_sat]
    
    
    
    
    
def normalize_input_for_dem(test_iter):
    #test_iter = torch.from_numpy(test_iter)
    input_images = test_iter.float()
    maxv = 822
    minv = -37
    #batch_size = input_images.shape[0]
    #cmin = torch.amin(input_images,(1,2)).reshape((batch_size,1,1))
    #cmax = torch.amax(input_images,(1,2)).reshape((batch_size,1,1))
    return (input_images-minv)/(maxv-minv) #(input_images-cmin)/(cmax-cmin)

class Normalize_range01:
    #bring to range 0 to 1

    def __init__(self, p=1):
        self.p = 1

    def __call__(self, x):
        result = normalize_input_for_dem(x)
        return result

    def __repr__(self):
        return "custom augmentation"

img_size = (96,96)
mask_resize = transforms.Resize(size = img_size, interpolation=Image.NEAREST)
#dem_resize = transforms.Resize(size = img_size)
data_transforms = transforms.Compose([#transforms.ToTensor(),
                                          transforms.Resize(size = img_size),
                                          Normalize_range01(),
                                          #transforms.ToPILImage(), 
                                          #transforms.RandomHorizontalFlip(),
                                          #transforms.RandomResizedCrop(size=100),
                                          #transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
                                          #transforms.RandomGrayscale(p=0.2),
                                          #transforms.GaussianBlur(kernel_size=9),
                                          #transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))
                                         ])

from torch.utils.data import random_split
import math
dataset = SwedishDataset(remove_list = remove_list, remove_water_list = remove_water_list)
#train_data, test_data = random_split(dataset, [math.ceil(len(dataset)*0.8), math.floor(len(dataset)*0.2)])
train_size = math.ceil(len(dataset)*0.9)
test_size = math.floor(len(dataset)*0.1)
train_dataset = torch.utils.data.Subset(dataset, range(train_size))
test_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))

print(len(train_dataset))
print(len(test_dataset))

print(dataset[4][0].shape)
dataset[4]


# In[ ]:


print('Starting to search for min and max')
maxv = 0
minv = 1000
for im,lab in dataset:
  if im.max()>maxv:
    maxv = im.max()

  if im.min()<minv:
    minv = im.min()
print(maxv, minv)
min_max = []
min_max.append(minv)
min_max.append(maxv)
print(maxv, minv)
with open('min_max_vals.npy', 'wb') as f:
    np.save(f, min_max)
    
print('Finished')


