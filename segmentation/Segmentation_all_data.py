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


    def convert_to_classes(self, temp_tensor):
          mask_tensor_1 = (temp_tensor == 111)|(temp_tensor == 112)|(temp_tensor == 113)|(temp_tensor == 114)|(temp_tensor== 115)|(temp_tensor == 116)|(temp_tensor == 117) 
          mask_tensor_2 = (temp_tensor == 118)
          mask_tensor_3 = (temp_tensor == 121)|(temp_tensor == 122)|(temp_tensor == 123)|(temp_tensor== 124)|(temp_tensor == 125)|(temp_tensor == 126)|(temp_tensor == 127)
          mask_tensor_4 = (temp_tensor == 128)
          mask_tensor_5 = (temp_tensor == 2)
          mask_tensor_6 = (temp_tensor == 3)
          mask_tensor_7 = (temp_tensor == 41)|(temp_tensor== 42)
          mask_tensor_8 = (temp_tensor == 51)|(temp_tensor == 52)|(temp_tensor == 53)
          mask_tensor_9 = (temp_tensor == 61)|(temp_tensor == 62)
          mask_tensor_outside = (temp_tensor==0)
          mask_tensors = [mask_tensor_1,mask_tensor_2,mask_tensor_3,mask_tensor_4,
                          mask_tensor_5,mask_tensor_6, mask_tensor_7,mask_tensor_8,
                          mask_tensor_9, mask_tensor_outside]
          temp_image = temp_tensor
          for i in range(10):
            temp_image = torch.where(mask_tensors[i], torch.tensor(i), temp_image)
          return temp_image
    
          
    def __getitem__(self, idx):
        img_path_dem, img_path_sat = self.data[idx]
        img_raster_dem = rasterio.open(img_path_dem).read()#[:,:96,:96]
        img_raster_sat = rasterio.open(img_path_sat).read()#[:,:170,:350]
        img_raster_sat = self.convert_to_classes(torch.from_numpy(img_raster_sat))
        
        if img_raster_dem[0].shape!=(100,100):
            if img_raster_dem[0][0,0]==0:
                img_raster_dem = img_raster_dem[:,1:,:]
            else:
                img_raster_dem = img_raster_dem[:,:-1,:]
        
        if self.transform:
          img_raster_dem = self.transform(torch.tensor(img_raster_dem))
        
        test = img_raster_sat
        outside_class = 9
        if torch.equal(test[0][0,:], torch.ones(len(test[0][0,:]))*outside_class):
            test = test[:,1:,:]
        if torch.equal(test[0][-1,:], torch.ones(len(test[0][-1,:]))*outside_class):
            test = test[:,:-1,:]
        if torch.equal(test[0][:,0], torch.ones(len(test[0][:,0]))*outside_class):
            test = test[:,:,1:]
        if torch.equal(test[0][:,-1], torch.ones(len(test[0][:,-1]))*outside_class):
            test = test[:,:,:-1]
        img_raster_sat = test
        img_raster_sat = mask_resize(img_raster_sat)
        
        return [img_raster_dem ,img_raster_sat[0].type(torch.LongTensor)]
    
    
    
    
    
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
dataset = SwedishDataset(transform = data_transforms, remove_list = remove_list, remove_water_list = remove_water_list)
#train_data, test_data = random_split(dataset, [math.ceil(len(dataset)*0.8), math.floor(len(dataset)*0.2)])
train_size = math.ceil(len(dataset)*0.9)
test_size = math.floor(len(dataset)*0.1)
train_dataset = torch.utils.data.Subset(dataset, range(train_size))
test_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))

print(len(train_dataset))
print(len(test_dataset))

print(dataset[4][0].shape)






class UNet(nn.Module):
    
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.contracting_11 = self.conv_block(in_channels=1, out_channels=64)
        self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_21 = self.conv_block(in_channels=64, out_channels=128)
        self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_31 = self.conv_block(in_channels=128, out_channels=256)
        self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_41 = self.conv_block(in_channels=256, out_channels=512)
        self.contracting_42 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle = self.conv_block(in_channels=512, out_channels=1024)
        self.expansive_11 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_12 = self.conv_block(in_channels=1024, out_channels=512)
        self.expansive_21 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_22 = self.conv_block(in_channels=512, out_channels=256)
        self.expansive_31 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_32 = self.conv_block(in_channels=256, out_channels=128)
        self.expansive_41 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_42 = self.conv_block(in_channels=128, out_channels=64)
        self.output = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1)
        
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels))
        return block
    
    def forward(self, X):
        contracting_11_out = self.contracting_11(X) # [-1, 64, 256, 256]
        contracting_12_out = self.contracting_12(contracting_11_out) # [-1, 64, 128, 128]
        contracting_21_out = self.contracting_21(contracting_12_out) # [-1, 128, 128, 128]
        contracting_22_out = self.contracting_22(contracting_21_out) # [-1, 128, 64, 64]
        contracting_31_out = self.contracting_31(contracting_22_out) # [-1, 256, 64, 64]
        contracting_32_out = self.contracting_32(contracting_31_out) # [-1, 256, 32, 32]
        contracting_41_out = self.contracting_41(contracting_32_out) # [-1, 512, 32, 32]
        contracting_42_out = self.contracting_42(contracting_41_out) # [-1, 512, 16, 16]
        middle_out = self.middle(contracting_42_out) # [-1, 1024, 16, 16]
        expansive_11_out = self.expansive_11(middle_out) # [-1, 512, 32, 32]
        expansive_12_out = self.expansive_12(torch.cat((expansive_11_out, contracting_41_out), dim=1)) # [-1, 1024, 32, 32] -> [-1, 512, 32, 32]
        expansive_21_out = self.expansive_21(expansive_12_out) # [-1, 256, 64, 64]
        expansive_22_out = self.expansive_22(torch.cat((expansive_21_out, contracting_31_out), dim=1)) # [-1, 512, 64, 64] -> [-1, 256, 64, 64]
        expansive_31_out = self.expansive_31(expansive_22_out) # [-1, 128, 128, 128]
        expansive_32_out = self.expansive_32(torch.cat((expansive_31_out, contracting_21_out), dim=1)) # [-1, 256, 128, 128] -> [-1, 128, 128, 128]
        expansive_41_out = self.expansive_41(expansive_32_out) # [-1, 64, 256, 256]
        expansive_42_out = self.expansive_42(torch.cat((expansive_41_out, contracting_11_out), dim=1)) # [-1, 128, 256, 256] -> [-1, 64, 256, 256]
        output_out = self.output(expansive_42_out) # [-1, num_classes, 256, 256]
        return output_out


# In[17]:


num_classes = 9
model = UNet(num_classes=num_classes)


# In[18]:


data_loader = DataLoader(train_dataset, batch_size=4)
print(len(train_dataset), len(data_loader))

X, Y = next(iter(data_loader))
print(X.shape, Y.shape)


# In[19]:


Y_pred = model(X)
print(Y_pred.shape)


# # Train

# In[22]:


batch_size = 32

epochs = 1000
lr = 0.001


# In[23]:


#dataset = CityscapeDataset(train_dir, label_model)
data_loader = DataLoader(train_dataset, batch_size=batch_size)


# In[24]:


model = UNet(num_classes=num_classes).to(device)


# In[25]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


# In[26]:


decayRate = 0.96
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)


# In[ ]:


step_losses = []
epoch_losses = []
for epoch in tqdm(range(epochs)):
    epoch_loss = 0
    for X, Y in tqdm(data_loader, total=len(data_loader), leave=False):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        Y_pred = model(X)
        loss = criterion(Y_pred, Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        step_losses.append(loss.item())
    my_lr_scheduler.step()
    epoch_losses.append(epoch_loss/len(data_loader))




model_name = "SwedenData/checkpoints/U-Net_normranged_normalized_all_data_1000epoch.pth"
torch.save(model.state_dict(), model_name)






