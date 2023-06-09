#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

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


device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print(device)


# In[2]:




# In[2]:


#with open('files_remove_indices/small_data_remove_list_indexes.npy', 'wb') as f:
#    np.save(f, images_containing_nine)

with open('files_remove_indices/small_data_remove_list_indexes.npy', 'rb') as f:
    remove_list = np.load(f)


# In[3]:


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
    #test_iter = torch.from_numpy(test_iter)
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


# # Model

# In[28]:


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


# In[98]:


num_classes = 9
model = UNet(num_classes=num_classes)


# # Wandb reporting

# In[24]:
import wandb

#!wandb login --relogin "40706c6c21e200af3bdb3840883f22e665e74441"
#wandb.login(key="40706c6c21e200af3bdb3840883f22e665e74441")

# # Training
# 

# In[29]:


import numpy as np
import torch
import torch.nn as nn


class SegmentationMetrics(object):
    r"""Calculate common metrics in semantic segmentation to evalueate model preformance.
    Supported metrics: Pixel accuracy, Dice Coeff, precision score and recall score.
    
    Pixel accuracy measures how many pixels in a image are predicted correctly.
    Dice Coeff is a measure function to measure similarity over 2 sets, which is usually used to
    calculate the similarity of two samples. Dice equals to f1 score in semantic segmentation tasks.
    
    It should be noted that Dice Coeff and Intersection over Union are highly related, so you need 
    NOT calculate these metrics both, the other can be calcultaed directly when knowing one of them.
    Precision describes the purity of our positive detections relative to the ground truth. Of all
    the objects that we predicted in a given image, precision score describes how many of those objects
    actually had a matching ground truth annotation.
    Recall describes the completeness of our positive predictions relative to the ground truth. Of
    all the objected annotated in our ground truth, recall score describes how many true positive instances
    we have captured in semantic segmentation.
    Args:
        eps: float, a value added to the denominator for numerical stability.
            Default: 1e-5
        average: bool. Default: ``True``
            When set to ``True``, average Dice Coeff, precision and recall are
            returned. Otherwise Dice Coeff, precision and recall of each class
            will be returned as a numpy array.
        ignore_background: bool. Default: ``True``
            When set to ``True``, the class will not calculate related metrics on
            background pixels. When the segmentation of background pixels is not
            important, set this value to ``True``.
        activation: [None, 'none', 'softmax' (default), 'sigmoid', '0-1']
            This parameter determines what kind of activation function that will be
            applied on model output.
    Input:
        y_true: :math:`(N, H, W)`, torch tensor, where we use int value between (0, num_class - 1)
        to denote every class, where ``0`` denotes background class.
        y_pred: :math:`(N, C, H, W)`, torch tensor.
    Examples::
        >>> metric_calculator = SegmentationMetrics(average=True, ignore_background=True)
        >>> pixel_accuracy, dice, precision, recall = metric_calculator(y_true, y_pred)
    """
    def __init__(self, eps=1e-5, average=True, ignore_background=True, activation='0-1'):
        self.eps = eps
        self.average = average
        self.ignore = ignore_background
        self.activation = activation

    @staticmethod
    def _one_hot(gt, pred, class_num):
        # transform sparse mask into one-hot mask
        # shape: (B, H, W) -> (B, C, H, W)
        input_shape = tuple(gt.shape)  # (N, H, W, ...)
        new_shape = (input_shape[0], class_num) + input_shape[1:]
        one_hot = torch.zeros(new_shape).to(pred.device, dtype=torch.float)
        target = one_hot.scatter_(1, gt.unsqueeze(1).long().data, 1.0)
        return target

    @staticmethod
    def _get_class_data(gt_onehot, pred, class_num):
        # perform calculation on a batch
        # for precise result in a single image, plz set batch size to 1
        matrix = np.zeros((3, class_num))

        # calculate tp, fp, fn per class
        for i in range(class_num):
            # pred shape: (N, H, W)
            class_pred = pred[:, i, :, :]
            # gt shape: (N, H, W), binary array where 0 denotes negative and 1 denotes positive
            class_gt = gt_onehot[:, i, :, :]

            pred_flat = class_pred.contiguous().view(-1, )  # shape: (N * H * W, )
            gt_flat = class_gt.contiguous().view(-1, )  # shape: (N * H * W, )

            tp = torch.sum(gt_flat * pred_flat)
            fp = torch.sum(pred_flat) - tp
            fn = torch.sum(gt_flat) - tp

            matrix[:, i] = tp.item(), fp.item(), fn.item()

        return matrix

    def _calculate_multi_metrics(self, gt, pred, class_num):
        # calculate metrics in multi-class segmentation
        matrix = self._get_class_data(gt, pred, class_num)
        if self.ignore:
            matrix = matrix[:, 1:]

        # tp = np.sum(matrix[0, :])
        # fp = np.sum(matrix[1, :])
        # fn = np.sum(matrix[2, :])

        pixel_acc = (np.sum(matrix[0, :]) + self.eps) / (np.sum(matrix[0, :]) + np.sum(matrix[1, :]))
        dice = (2 * matrix[0] + self.eps) / (2 * matrix[0] + matrix[1] + matrix[2] + self.eps)
        precision = (matrix[0] + self.eps) / (matrix[0] + matrix[1] + self.eps)
        recall = (matrix[0] + self.eps) / (matrix[0] + matrix[2] + self.eps)

        if self.average:
            dice = np.average(dice)
            precision = np.average(precision)
            recall = np.average(recall)

        return pixel_acc, dice, precision, recall

    def __call__(self, y_true, y_pred):
        class_num = y_pred.size(1)

        if self.activation in [None, 'none']:
            activation_fn = lambda x: x
            activated_pred = activation_fn(y_pred)
        elif self.activation == "sigmoid":
            activation_fn = nn.Sigmoid()
            activated_pred = activation_fn(y_pred)
        elif self.activation == "softmax":
            activation_fn = nn.Softmax(dim=1)
            activated_pred = activation_fn(y_pred)
        elif self.activation == "0-1":
            pred_argmax = torch.argmax(y_pred, dim=1)
            activated_pred = self._one_hot(pred_argmax, y_pred, class_num)
        else:
            raise NotImplementedError("Not a supported activation!")

        gt_onehot = self._one_hot(y_true, y_pred, class_num)
        pixel_acc, dice, precision, recall = self._calculate_multi_metrics(gt_onehot, activated_pred, class_num)
        return pixel_acc, dice, precision, recall


class BinaryMetrics():
    r"""Calculate common metrics in binary cases.
    In binary cases it should be noted that y_pred shape shall be like (N, 1, H, W), or an assertion 
    error will be raised.
    Also this calculator provides the function to calculate specificity, also known as true negative 
    rate, as specificity/TPR is meaningless in multiclass cases.
    """
    def __init__(self, eps=1e-5, activation='0-1'):
        self.eps = eps
        self.activation = activation

    def _calculate_overlap_metrics(self, gt, pred):
        output = pred.view(-1, )
        target = gt.view(-1, ).float()

        tp = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1 - target))  # FP
        fn = torch.sum((1 - output) * target)  # FN
        tn = torch.sum((1 - output) * (1 - target))  # TN

        pixel_acc = (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps)
        dice = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)
        precision = (tp + self.eps) / (tp + fp + self.eps)
        recall = (tp + self.eps) / (tp + fn + self.eps)
        specificity = (tn + self.eps) / (tn + fp + self.eps)

        return pixel_acc, dice, precision, specificity, recall

    def __call__(self, y_true, y_pred):
        # y_true: (N, H, W)
        # y_pred: (N, 1, H, W)
        if self.activation in [None, 'none']:
            activation_fn = lambda x: x
            activated_pred = activation_fn(y_pred)
        elif self.activation == "sigmoid":
            activation_fn = nn.Sigmoid()
            activated_pred = activation_fn(y_pred)
        elif self.activation == "0-1":
            sigmoid_pred = nn.Sigmoid()(y_pred)
            activated_pred = (sigmoid_pred > 0.5).float().to(y_pred.device)
        else:
            raise NotImplementedError("Not a supported activation!")

        assert activated_pred.shape[1] == 1, 'Predictions must contain only one channel' \
                                             ' when performing binary segmentation'
        pixel_acc, dice, precision, specificity, recall = self._calculate_overlap_metrics(y_true.to(y_pred.device,
                                                                                                    dtype=torch.float),
                                                                                          activated_pred)
        return [pixel_acc, dice, precision, specificity, recall]
    
metric_calculator = SegmentationMetrics(average=True, ignore_background=True)


# In[30]:


# # Train


batch_size = 32

epochs = 1000
lr = 0.0001



#dataset = CityscapeDataset(train_dir, label_model)
data_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
testdata_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)


# In[24]:


model = UNet(num_classes=num_classes).to(device)


# In[25]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


# In[26]:


decayRate = 0.94
#my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.9)


######################################
import wandb
import random
from sklearn.metrics import classification_report

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Segmentation_unet_smalldata",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "UNET",
    "dataset": "Sweden_small",
    "epochs": epochs,
    }
)
log_every = 50
test_every = 50
######################################

step_losses = []
epoch_losses = []
for epoch in tqdm(range(epochs)):
    epoch_loss = 0
    Y_list = []
    Y_pred_N_list = []
    Y_pred_list = []
    
    for X, Y in tqdm(data_loader, total=len(data_loader), leave=False):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        Y_pred_N = model(X)
        loss = criterion(Y_pred_N, Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        step_losses.append(loss.item())
        wandb.log({'iteration_loss': loss.item()})
        if epoch % log_every==0:
            Y_list.append(Y)
            Y_pred_N_list.append(Y_pred_N)
            Y_pred_list.append(torch.argmax(Y_pred_N, dim=1))

    my_lr_scheduler.step()
    epoch_losses.append(epoch_loss/len(data_loader))
    current_lr = my_lr_scheduler.get_last_lr()[0]
    
    
    # log metrics to wandb
    wandb.log({"epoch_loss": epoch_loss, "learning_rate": current_lr})
    if epoch % log_every==0:
        model_name = "SwedenData/checkpoints/U-Net_normranged_normalized_smalldata_1000epoch_lr0001_4.pth"
        torch.save(model.state_dict(), model_name)
        pixel_accuracy, dice, precision, recall = metric_calculator(torch.vstack(Y_list), torch.vstack(Y_pred_N_list))
        wandb.log({"epoch_pixel_acc": pixel_accuracy, "epoch_dice":dice,
                   "epoch_precision":precision, "epoch_recall":recall})

        y_preds_flat = torch.vstack(Y_pred_list).data.cpu().numpy().reshape(1,-1)[0]
        y_trues_flat = torch.vstack(Y_list).data.cpu().numpy().reshape(1,-1)[0]
        report = classification_report(y_preds_flat, y_trues_flat,labels=[0, 1, 2, 3,4,5,6,7,8], output_dict = True)
        wandb.log({"sckt_epoch_acc": report['accuracy'],
                   "sckt_epoch_macro_precision": report['macro avg']['precision'], 
                   "sckt_epoch_macro_recall": report['macro avg']['recall'],
                   "sckt_epoch_macro_f1-score": report['macro avg']['f1-score'],
                   "sckt_epoch_weighted_precision": report['weighted avg']['precision'],
                   "sckt_epoch_weighted_recall": report['weighted avg']['recall'],
                   "sckt_epoch_weighted_f1-score": report['weighted avg']['f1-score']})

    if epoch % test_every==0:
        model.eval()
        Y_list = []
        Y_pred_N_list = []
        Y_pred_list = []
        with torch.no_grad():
            for X,Y in testdata_loader:
                X, Y = X.to(device), Y.to(device)
                Y_pred_N = model(X)
                Y_pred = torch.argmax(Y_pred_N, dim=1)
                Y_list.append(Y)
                Y_pred_N_list.append(Y_pred_N)
                Y_pred_list.append(Y_pred)


        y_preds_flat = torch.vstack(Y_pred_list).data.cpu().numpy().reshape(1,-1)[0]
        y_trues_flat = torch.vstack(Y_list).data.cpu().numpy().reshape(1,-1)[0]
        report = classification_report(y_preds_flat, y_trues_flat,labels=[0, 1, 2, 3,4,5,6,7,8], output_dict = True)
        wandb.log({"testeval_sckt_epoch_acc": report['accuracy'],
                   "testeval_sckt_epoch_macro_precision": report['macro avg']['precision'], 
                   "testeval_sckt_epoch_macro_recall": report['macro avg']['recall'],
                   "testeval_sckt_epoch_macro_f1-score": report['macro avg']['f1-score'],
                   "testeval_sckt_epoch_weighted_precision": report['weighted avg']['precision'],
                   "testeval_sckt_epoch_weighted_recall": report['weighted avg']['recall'],
                   "testeval_sckt_epoch_weighted_f1-score": report['weighted avg']['f1-score']})

        Y_list = []
        Y_pred_N_list = []
        Y_pred_list = []
        with torch.no_grad():
            for X,Y in data_loader:
                X, Y = X.to(device), Y.to(device)
                Y_pred_N = model(X)
                Y_pred = torch.argmax(Y_pred_N, dim=1)
                Y_list.append(Y)
                Y_pred_N_list.append(Y_pred_N)
                Y_pred_list.append(Y_pred)


        y_preds_flat = torch.vstack(Y_pred_list).data.cpu().numpy().reshape(1,-1)[0]
        y_trues_flat = torch.vstack(Y_list).data.cpu().numpy().reshape(1,-1)[0]
        report = classification_report(y_preds_flat, y_trues_flat,labels=[0, 1, 2, 3,4,5,6,7,8], output_dict = True)
        wandb.log({"traineval_sckt_epoch_acc": report['accuracy'],
                   "traineval_sckt_epoch_macro_precision": report['macro avg']['precision'], 
                   "traineval_sckt_epoch_macro_recall": report['macro avg']['recall'],
                   "traineval_sckt_epoch_macro_f1-score": report['macro avg']['f1-score'],
                   "traineval_sckt_epoch_weighted_precision": report['weighted avg']['precision'],
                   "traineval_sckt_epoch_weighted_recall": report['weighted avg']['recall'],
                   "traineval_sckt_epoch_weighted_f1-score": report['weighted avg']['f1-score']})



        model.train()
                        
    


wandb.finish()


print('Finished training_____________________')

#Evaluation of model after loading it
model_path = "SwedenData/checkpoints/U-Net_normranged_normalized_smalldata_1000epoch_lr0001_4.pth"
model_ = UNet(num_classes=num_classes).to(device)
model_.load_state_dict(torch.load(model_path))
model_.eval()


Y_list = []
Y_pred_N_list = []
Y_pred_list = []
with torch.no_grad():
    for X,Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        Y_pred_N = model_(X)
        Y_pred = torch.argmax(Y_pred_N, dim=1)
        Y_list.append(Y)
        Y_pred_N_list.append(Y_pred_N)
        Y_pred_list.append(Y_pred)

pixel_accuracy, dice, precision, recall = metric_calculator(torch.vstack(Y_list), torch.vstack(Y_pred_N_list))
print("Pixel accuracy: {:.2f}".format(pixel_accuracy))
print("Dice: {:.2f}".format(dice))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
f_score = 2*precision*recall/(precision+recall)
print('F-score: {:.2f}'.format(f_score))


from sklearn.metrics import classification_report
y_preds_flat = torch.vstack(Y_pred_list).data.cpu().numpy().reshape(1,-1)[0]
y_trues_flat = torch.vstack(Y_list).data.cpu().numpy().reshape(1,-1)[0]
report = classification_report(y_preds_flat, y_trues_flat,labels=[0, 1, 2, 3,4,5,6,7,8])
print(report)


print('TEST SET EVALUATION_______________')
Y_list = []
Y_pred_N_list = []
Y_pred_list = []
with torch.no_grad():
    for X,Y in testdata_loader:
        X, Y = X.to(device), Y.to(device)
        Y_pred_N = model_(X)
        Y_pred = torch.argmax(Y_pred_N, dim=1)
        Y_list.append(Y)
        Y_pred_N_list.append(Y_pred_N)
        Y_pred_list.append(Y_pred)

pixel_accuracy, dice, precision, recall = metric_calculator(torch.vstack(Y_list), torch.vstack(Y_pred_N_list))
print("Pixel accuracy: {:.2f}".format(pixel_accuracy))
print("Dice: {:.2f}".format(dice))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
f_score = 2*precision*recall/(precision+recall)
print('F-score: {:.2f}'.format(f_score))


from sklearn.metrics import classification_report
y_preds_flat = torch.vstack(Y_pred_list).data.cpu().numpy().reshape(1,-1)[0]
y_trues_flat = torch.vstack(Y_list).data.cpu().numpy().reshape(1,-1)[0]
report = classification_report(y_preds_flat, y_trues_flat,labels=[0, 1, 2, 3,4,5,6,7,8])
print(report)





