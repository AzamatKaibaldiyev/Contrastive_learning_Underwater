#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = 1


## Standard libraries
import os
from copy import deepcopy

## Imports for plotting
import matplotlib.pyplot as plt
#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('svg', 'pdf') # For export
import matplotlib

#import seaborn as sns
#sns.set()

## tqdm for loading bars
from tqdm.notebook import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

## Torchvision
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image

# PyTorch Lightning
import pytorch_lightning as pl
#try:
#    import pytorch_lightning as pl
#except ModuleNotFoundError: # Google Colab does not have PyTorch Lightning installed by default. Hence, we do it here if necessary
#    !pip install --quiet pytorch-lightning>=1.4
#    import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint



# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
#DATASET_PATH = "../data"
# Path to the folder where the pretrained models are saved
#CHECKPOINT_PATH = '/home/azamat.kaibaldiyev/Tasmania_ohara_labels/checkpoints_images/'
CHECKPOINT_PATH = '/home/azamat.kaibaldiyev/Tasmania_ohara_labels/checkpoints_Ohara_images_and_bathym_doublenet_32/'

# In this notebook, we use data loaders with heavier computational processing. It is recommended to use as many
# workers as possible in a data loader, which corresponds to the number of CPU cores
NUM_WORKERS = os.cpu_count()

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
print("Number of workers:", NUM_WORKERS)


# In[2]:


#all images: from professor
import glob
imgs = sorted(glob.glob('Tasmania_ohara_labels/Ohara_images_from_prof/*/*'))
print(len(imgs))

# In[3]:


bathym =  sorted(glob.glob('Tasmania_ohara_labels/cropped_dems_32/*'))
print(len(bathym))


bathym_imgs =  sorted(glob.glob('Tasmania_ohara_labels/cropped_dems_32_imgs_100/*'))
print(len(bathym_imgs))


########################################################################################################################

imgs_new = sorted(glob.glob('Tasmania_ohara_labels/Ohara_images_from_prof_2/*/*'))
print(len(imgs_new))

imgs = imgs+imgs_new
print(len(imgs))


# Python program to store list to JSON file
import json

# def write_list(a_list):
#     print("Started writing list data into a json file")
#     with open("files_remove_indices/Remove_paths_with_nan_dem16.json", "w") as fp:
#         json.dump(a_list, fp)
#         print("Done writing JSON data into .json file")
#write_list(remove_bath_paths)

# Read list to memory
def read_list():
    # for reading also binary mode is important
    with open('files_remove_indices/Remove_paths_with_nan_dem32.json', 'rb') as fp:
        n_list = json.load(fp)
        return n_list

# assume you have the following list
remove_bath_paths = read_list()
print(len(remove_bath_paths))


bathym_imgs_stay = [ele for ele in bathym_imgs if ele not in remove_bath_paths]
print(len(bathym_imgs_stay))
stay_idxs_imgs = [i for i,ele in enumerate(bathym_imgs) if ele not in remove_bath_paths]
print(len(stay_idxs_imgs ))
imgs_to_stay= [ele for i, ele in enumerate(imgs) if i in stay_idxs_imgs]
print(len(imgs_to_stay))

bathym_and_imgs_path = list(zip(bathym_imgs_stay,imgs_to_stay[:len(bathym_imgs_stay)]))

print('Bathym and images len: ', len(bathym_and_imgs_path))

########################################################################################################################



# In[51]:


from PIL import Image
from torch.utils.data import Dataset, DataLoader
import glob
import rasterio

dem_size = 32
img_size = 100

class SwedishDataset(Dataset):
    def __init__(self,  transform_dem  = None, transform_img = None, dataset_path = None, labels = False):
        self.imgs_path = dataset_path
        
        if labels:
            self.file_list_dem = df_merged_images['Left image name']

        self.transform_img = transform_img
        self.transform_dem = transform_dem
        self.labels = labels
        
    def __len__(self):
        return len(self.imgs_path)

          
    def __getitem__(self, idx):
        img_path_dem = self.imgs_path[idx]
        
        dem_raster = Image.open(img_path_dem[0])
        #img_raster = torch.from_numpy(np.array(Image.open(img_path_dem[1])))
        img_raster = Image.open(img_path_dem[1])
        dem_raster = self.transform_dem(dem_raster)
        img_raster = self.transform_img(img_raster)
        
#         if len(img_path_dem)==2:
#             dem_raster = Image.open(img_path_dem[0])
#             img_raster = np.array(Image.open(img_path_dem[1]))[:,:,0]
#             dem_raster = dem_transforms(dem_raster)
#             img_raster = img_transforms(img_raster)
#             img_raster_dem = [dem_raster, img_raster]
#         else:
#             img_raster_dem = Image.open(img_path_dem)
#             img_raster_dem = self.transform(img_raster_dem)
        
#         if self.labels:
#             label = int(df_merged_images['cluster_id'][idx])
#         else:
#             label = 'no_label'
        
        return [dem_raster,img_raster]
    

def normalize_input_for_dem(test_iter):
    #test_iter = torch.from_numpy(test_iter)
    input_images = test_iter.float()
    maxv = 104
    minv = 5
#     batch_size = input_images.shape[0]
#     cmin = torch.amin(input_images,(1,2)).reshape((batch_size,1,1))
#     cmax = torch.amax(input_images,(1,2)).reshape((batch_size,1,1))
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

    
class ContrastiveTransformations(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]
    
    


contrast_transforms_dem = transforms.Compose([transforms.ToTensor(),
                                          Normalize_range01(),
                                          transforms.Resize(size = dem_size),
                                          #Normalize_range01(),
                                          #transforms.ToPILImage(), 
                                          #transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop(size=dem_size),
                                          #transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
                                          #transforms.RandomGrayscale(p=0.2),
                                          #transforms.GaussianBlur(kernel_size=9),
                                          #transforms.ToTensor(),
                                          #transforms.Normalize((0.5,), (0.5,))                                         
                                         ])

contrast_transforms_img = transforms.Compose([transforms.ToTensor(),
#                                           Normalize_range01(),
                                          transforms.Resize(size = img_size),
                                          #Normalize_range01(),
                                          #transforms.ToPILImage(), 
                                          #transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop(size=img_size),
                                          #transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
                                          #transforms.RandomGrayscale(p=0.2),
                                          #transforms.GaussianBlur(kernel_size=9),
                                          #transforms.ToTensor(),
                                          #transforms.Normalize((0.5,), (0.5,))                                         
                                         ])

# dem_transforms = transforms.Compose([transforms.ToTensor(),
#                                     Normalize_range01(),
#                                     transforms.Resize(size = dem_size),
#                                     transforms.RandomResizedCrop(size=dem_size),
#                                     #transforms.Normalize((0.5,), (0.5,))
#                                     ])

# img_transforms = transforms.Compose([transforms.ToTensor(),
#                                      transforms.Resize(size = 100),
#                                      transforms.RandomResizedCrop(size=dem_size),
#                                      #transforms.Normalize((0.5,), (0.5,))
#                                      ])

from torch.utils.data import random_split
import math
dataset_path = bathym_and_imgs_path
dataset = SwedishDataset(transform_dem = ContrastiveTransformations(contrast_transforms_dem, n_views = 2),
                         transform_img = ContrastiveTransformations(contrast_transforms_img, n_views = 2),
                         dataset_path = dataset_path)

print(len(dataset))
dataset_len =len(dataset)

from sklearn.model_selection import ShuffleSplit
rs = ShuffleSplit(n_splits=1, test_size=.1, random_state=0)
for i, (train_index, test_index) in enumerate(rs.split(dataset)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")

train_set_indexes = train_index
test_set_indexes = test_index

# train_set_indexes = np.arange(200)
# test_set_indexes = np.arange(200)

train_dataset = torch.utils.data.Subset(dataset, train_set_indexes)
test_dataset = torch.utils.data.Subset(dataset, test_set_indexes)

print(len(train_dataset))
print(len(test_dataset))




import wandb
import random
from sklearn.metrics import classification_report

epochs = 300
lr = 1e-5
#wandb.init(settings=wandb.Settings(start_method="fork"))
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Contrastive_SimCLR_Ohara_doublenet",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "SimCLR",
    "dataset": "Ohara_dataset_dems_and_images_56000",
    "epochs": epochs,
    }
)


# In[47]:


class SimCLR(pl.LightningModule):

    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, 'The temperature must be a positive float!'
        
        
        # Base model1 f(.)
        self.convnet_dem = torchvision.models.resnet18(num_classes=4*hidden_dim)  # Output of last linear layer
        #model = torchvision.models.resnet18()
        num_input_channel_dem = 1
        self.convnet_dem.conv1 = nn.Conv2d(num_input_channel_dem, 64, kernel_size=7, stride=2, padding=3,bias=False)
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet_dem.fc = nn.Sequential(
            self.convnet_dem.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4*hidden_dim, hidden_dim)
        )
        
        # Base model2 f(.)
        self.convnet_img = torchvision.models.resnet18(num_classes=4*hidden_dim)  # Output of last linear layer
        #model = torchvision.models.resnet18()
        num_input_channel_img = 3
        self.convnet_img.conv1 = nn.Conv2d(num_input_channel_img, 64, kernel_size=7, stride=2, padding=3,bias=False)
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet_img.fc = nn.Sequential(
            self.convnet_img.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4*hidden_dim, hidden_dim)
        )
        
        

    def configure_optimizers(self):
        optimizer_dem = optim.AdamW(self.convnet_dem.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        
        optimizer_img = optim.AdamW(self.convnet_dem.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        
        lr_scheduler_dem = optim.lr_scheduler.CosineAnnealingLR(optimizer_dem,
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr/50)
        lr_scheduler_img = optim.lr_scheduler.CosineAnnealingLR(optimizer_img,
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr/50)
        
        return [optimizer_dem, optimizer_img], [lr_scheduler_dem,lr_scheduler_img]

    def info_nce_loss(self, model, model_type, batch, mode='train'):
        imgs= batch
        imgs = torch.cat(imgs, dim=0)

        # Encode all images
        feats = model(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(model_type+mode+'_loss', nll)
        wandb.log({model_type+'SimCLR_'+mode+'_loss':nll})
        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)],
                             dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        wandb.log({model_type+'SimCLR_'+mode+'_acc_top1':(sim_argsort == 0).float().mean()})
        wandb.log({model_type+'SimCLR_'+mode+'_acc_top5':(sim_argsort < 5).float().mean()})
        wandb.log({model_type+'SimCLR_'+mode+'_acc_mean_pos': 1+sim_argsort.float().mean()})
        
        self.log(model_type+mode+'_acc_top1', (sim_argsort == 0).float().mean())
        self.log(model_type+mode+'_acc_top5', (sim_argsort < 5).float().mean())
        self.log(model_type+mode+'_acc_mean_pos', 1+sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx, optimizer_idx):

        if optimizer_idx == 0:
            dems, imgs = batch
            loss_dem = self.info_nce_loss(self.convnet_dem,'dem_', dems, mode='train')
            loss_img = self.info_nce_loss(self.convnet_img,'img_', imgs, mode='train')
            total_loss = loss_dem+loss_img
            return total_loss
        
        elif optimizer_idx == 1:
            dems, imgs = batch
            loss_dem = self.info_nce_loss(self.convnet_dem,'dem_', dems, mode='train')
            loss_img = self.info_nce_loss(self.convnet_img,'img_', imgs, mode='train')
            total_loss = loss_dem+loss_img
            return total_loss

    def validation_step(self, batch, batch_idx):
        dems, imgs = batch
        self.info_nce_loss(self.convnet_dem,'dem_', dems, mode='val')
        self.info_nce_loss(self.convnet_img,'img_', imgs, mode='val')
#         self.info_nce_loss(batch, mode='val')
        
    def forward(self, x):
        if x[0].shape[0]==1:
            return self.convnet_dem(x)
        else:
            return self.convnet_img(x)
        
    


# In[10]:


def train_simclr(batch_size, max_epochs=500, **kwargs):
    trainer = pl.Trainer(log_every_n_steps=10, default_root_dir=os.path.join(CHECKPOINT_PATH, 'SimCLR'),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='dem_val_acc_top5'),
                                    LearningRateMonitor('epoch')])
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, 'SimCLR.ckpt')
    if os.path.isfile(pretrained_filename):
        print(f'Found pretrained model at {pretrained_filename}, loading...')
        model = SimCLR.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
    else:
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                       drop_last=True, pin_memory=True, num_workers=2)
        val_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                     drop_last=False, pin_memory=True, num_workers=2)
        pl.seed_everything(42) # To be reproducable
        model = SimCLR(max_epochs=max_epochs, **kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    return model


# In[11]:


simclr_model = train_simclr(batch_size=256,
                            hidden_dim=128,
                            lr=5e-4,
                            temperature=0.07,
                            weight_decay=1e-4,
                            max_epochs=300)





