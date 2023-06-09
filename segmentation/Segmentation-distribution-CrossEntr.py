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


# # Define dataset

# In[3]:


with open('files_remove_indices/small_data_remove_list_indexes.npy', 'rb') as f:
    remove_list = np.load(f)


# In[4]:


from torch.utils.data import Dataset, DataLoader
import glob
import rasterio
class SwedishDataset(Dataset):
    def __init__(self,  transform  = None, remove_list = None):
        self.imgs_path = "SwedenData/100tiles/"
        file_list_dem1 = sorted(glob.glob(self.imgs_path + 'dem/'+ "*/*"))
        file_list_sat1 = sorted(glob.glob(self.imgs_path + 'sat/'+ "*/*"))

        self.data = []
        for img_path_dem,img_path_sat in zip(file_list_dem1,file_list_sat1):
          self.data.append([img_path_dem,img_path_sat])

        self.transform = transform
        #self.dems1_list = file_list_dem1 
        #self.sat1_list = file_list_sat1
        if remove_list is not None:
            self.data = np.array(self.data)
            mask = np.ones(len(self.data), dtype=bool)
            mask[remove_list] = False
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

    input_images = test_iter.float()
    #maxv = 822
    #minv = -37
    batch_size = input_images.shape[0]
    cmin = torch.amin(input_images,(1,2)).reshape((batch_size,1,1))
    cmax = torch.amax(input_images,(1,2)).reshape((batch_size,1,1))
    return (input_images-cmin)/(cmax-cmin) #(input_images-minv)/(maxv-minv) #(input_images-cmin)/(cmax-cmin)

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
dataset = SwedishDataset(transform = data_transforms, remove_list = remove_list)

with open('files_remove_indices/train_set_indexes.npy', 'rb') as f:
    train_set_indexes = np.load(f)

with open('files_remove_indices/test_set_indexes.npy', 'rb') as f:
    test_set_indexes = np.load(f)


train_dataset = torch.utils.data.Subset(dataset, train_set_indexes)
test_dataset = torch.utils.data.Subset(dataset, test_set_indexes)
print(len(train_dataset))
print(len(test_dataset))

print(dataset[4][0].shape)
dataset[4]


# In[6]:

batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

def apply_along_axis(function, x, axis,minlength: int = 0):
     return torch.stack([
                        function(x_i,minlength=minlength) for x_i in torch.unbind(x, dim=axis)
                        ], dim=axis)


# In[8]:


apply_along_axis(torch.bincount,next(iter(train_dataloader))[1].flatten(start_dim=1), 0 ,minlength = 9)/9216



# Use the torchvision's implementation of ResNeXt, but add FC layer for a different number of classes (27) and a Sigmoid instead of a default Softmax.
import torchvision
class Resnet18(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        #resnet = torchvision.models.resnext50_32x4d(pretrained=True)
        resnet = torchvision.models.resnet18()
        
        num_input_channel = 1
        resnet.conv1 = nn.Conv2d(num_input_channel, 64, kernel_size=7, stride=2, padding=3,bias=False)
        #resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        )
        self.base_model = resnet
        
 
    def forward(self, x):
        return self.base_model(x)
 
# Initialize the model
model = Resnet18(n_classes = 9)
# Switch model to the training mode
model.train()
model.to(device)


# In[8]:


#criterion = nn.BCELoss()
#criterion = nn.CrossEntropyLoss()
criterion = torch.nn.BCEWithLogitsLoss()

# In[9]:


max_epoch_number = 1000
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
log_every = 50
test_every = 50
save_checkpoint = "SwedenData/checkpoints/Segmentation_distribution_normranged_normalized_smalldata_1000epoch_celoss_local_bc64.pth"


# In[ ]:

#wandb.login(key="40706c6c21e200af3bdb3840883f22e665e74441")


import wandb
import random
import sklearn

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Segmentation_distribution_smalldata_crossentr",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "resnet18",
    "optimizer": "CELoss",
    "dataset": "Sweden_small",
    "epochs": max_epoch_number,
    }
)





from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score,explained_variance_score

epoch = 0
iteration = 0
softmax = nn.Softmax(dim=1)
model.train()
while True:
    batch_losses = []
    Y_list = []
    Y_pred_N_list = []

    for imgs, targets in train_dataloader:
        targets = apply_along_axis(torch.bincount,targets.flatten(start_dim=1), 0 ,minlength = 9)/9216
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
 
        model_result = model(imgs)
        loss = criterion(model_result, targets.type(torch.float))
 
        batch_loss_value = loss.item()
        loss.backward()
        optimizer.step()
 
        batch_losses.append(batch_loss_value)
        wandb.log({'iteration_loss': loss.item()})
 
        iteration += 1

        if epoch % log_every==0:
            Y_list.append(targets)
            Y_pred_N_list.append(softmax(model_result))

    
    my_lr_scheduler.step()
    current_lr = my_lr_scheduler.get_last_lr()[0]
    loss_value = np.mean(batch_losses)
    wandb.log({"epoch_loss": loss_value, "learning_rate": current_lr})



    if epoch % log_every==0:
        model_name = save_checkpoint
        torch.save(model.state_dict(), model_name)

        kl_div = F.kl_div(torch.vstack(Y_pred_N_list).log(), torch.vstack(Y_list), None, None, 'batchmean')
        wandb.log({"train_KL_divergence": kl_div})

        msle = []#(the best value is 0.0)
        mse = []#(the best value is 0.0)
        r2 = []#Best possible score is 1.0 and it can be negative
        evs = []#Best possible score is 1.0, lower values are worse.
        with torch.no_grad():
            for y_true,y_pred in zip(torch.vstack(Y_list).cpu().numpy(), torch.vstack(Y_pred_N_list).cpu().numpy()):
                msle.append(mean_squared_log_error(y_true, y_pred))
                mse.append(mean_squared_error(y_true, y_pred))
                r2.append(r2_score(y_true, y_pred))
                evs.append(explained_variance_score(y_true, y_pred))
            wandb.log({"train_mean_squared_log_error": np.mean(msle),
                      "train_mean_squared_error": np.mean(mse),
                      "train_r2_score": np.mean(r2),
                      "train_explained_variance_score": np.mean(evs)})



    if epoch % test_every==0:
        model.eval()

        Y_list = []
        Y_pred_N_list = []
        with torch.no_grad():
            for X,Y in train_dataloader:
                X, Y = X.to(device), Y.to(device)
                Y_pred_N = model(X)
                targets = apply_along_axis(torch.bincount,Y.flatten(start_dim=1), 0 ,minlength = 9)/9216
                Y_list.append(targets)
                Y_pred_N_list.append(softmax(Y_pred_N))

        kl_div = F.kl_div(torch.vstack(Y_pred_N_list).log(), torch.vstack(Y_list), None, None, 'batchmean')
        wandb.log({"traineval_KL_divergence": kl_div})

        msle = []#(the best value is 0.0)
        mse = []#(the best value is 0.0)
        r2 = []#Best possible score is 1.0 and it can be negative
        evs = []#Best possible score is 1.0, lower values are worse.
        with torch.no_grad():
            for y_true,y_pred in zip(torch.vstack(Y_list).cpu().numpy(), torch.vstack(Y_pred_N_list).cpu().numpy()):
                msle.append(mean_squared_log_error(y_true, y_pred))
                mse.append(mean_squared_error(y_true, y_pred))
                r2.append(r2_score(y_true, y_pred))
                evs.append(explained_variance_score(y_true, y_pred))
            wandb.log({"traineval_mean_squared_log_error": np.mean(msle),
                      "traineval_mean_squared_error": np.mean(mse),
                      "traineval_r2_score": np.mean(r2),
                      "traineval_explained_variance_score": np.mean(evs)})


        Y_list = []
        Y_pred_N_list = []
        with torch.no_grad():
            for X,Y in test_dataloader:
                X, Y = X.to(device), Y.to(device)
                Y_pred_N = model(X)
                targets = apply_along_axis(torch.bincount,Y.flatten(start_dim=1), 0 ,minlength = 9)/9216
                Y_list.append(targets)
                Y_pred_N_list.append(softmax(Y_pred_N))

        kl_div = F.kl_div(torch.vstack(Y_pred_N_list).log(), torch.vstack(Y_list), None, None, 'batchmean')
        wandb.log({"testeval_KL_divergence": kl_div})

        msle = []#(the best value is 0.0)
        mse = []#(the best value is 0.0)
        r2 = []#Best possible score is 1.0 and it can be negative
        evs = []#Best possible score is 1.0, lower values are worse.
        with torch.no_grad():
            for y_true,y_pred in zip(torch.vstack(Y_list).cpu().numpy(), torch.vstack(Y_pred_N_list).cpu().numpy()):
                msle.append(mean_squared_log_error(y_true, y_pred))
                mse.append(mean_squared_error(y_true, y_pred))
                r2.append(r2_score(y_true, y_pred))
                evs.append(explained_variance_score(y_true, y_pred))
            wandb.log({"testeval_mean_squared_log_error": np.mean(msle),
                      "testeval_mean_squared_error": np.mean(mse),
                      "testeval_r2_score": np.mean(r2),
                      "testeval_explained_variance_score": np.mean(evs)})


        model.train()





model_name = save_checkpoint
torch.save(model.state_dict(), model_name)





