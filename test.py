#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:14:17 2019

@author: chenzj
"""


import sys

from datasets import DatasetTest 
#import lovasz_losses

sys.path.append("/home/chenzj/ISIC/segmentation/models")
from cenet import CE_Net_ , CE_Net_backbone_DAC_with_inception



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image


torch.manual_seed(1)           
torch.cuda.manual_seed(1)      


parser = argparse.ArgumentParser(description='Transferlearning...')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', default='ckptv1.t7', type=str, help='model to be test')
parser.add_argument('--aimdir', default='ckptv1', type=str, help='model to be test')
args = parser.parse_args()




device = 'cuda' if torch.cuda.is_available() else 'cpu'
#best_acc = 0  # best test accuracy
best_iou = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data Prepare
print('==> Preparing data..')

transform = transforms.Compose([
    transforms.RandomChoice([
        transforms.RandomRotation(90),
        transforms.RandomRotation(180),
        transforms.RandomRotation(270),
    ]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
'''
    transforms.RandomChoice([
        transforms.ColorJitter(brightness=0.2),
        transforms.ColorJitter(contrast=0.2), 
        transforms.ColorJitter(saturation=0.2),
        transforms.ColorJitter(hue=0.15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), 
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), 
    ]),
'''
])

transform_img = transforms.Compose([
    transforms.RandomChoice([
        transforms.ColorJitter(brightness=0.1),
        transforms.ColorJitter(contrast=0.2), 
        transforms.ColorJitter(saturation=0.1),
        transforms.ColorJitter(hue=0.15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), 
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), 
    ]),

    transforms.ToTensor(),

    transforms.Normalize(( 0.7082, 0.5715, 0.5273), 
                         ( 0.0981, 0.1134, 0.1276)),
]) 
    


test_dataset = DatasetTest(img_path='/home/chenzj/ISIC/Data2018/ISIC2018_Task1-2_Test_Input/',  
                           transform=transform_img)



testloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=1)



# Model
print('==> Building model..')

#net = CE_Net_().to(device)
net = CE_Net_backbone_DAC_with_inception().to(device)

#print(net)

if device == 'cuda':
    cudnn.benchmark = True



# Load trained model.
print('==> trained model..')
assert os.path.isdir('predict'), 'Error: no trained model directory found!'
checkpoint = torch.load(os.path.join('./predict/', args.model))
net.load_state_dict(checkpoint['net'])
best_iou = checkpoint['IOU']

end_epoch = checkpoint['epoch']

print(best_iou)
print(end_epoch)

if not os.path.isdir(os.path.join('./predict/', args.aimdir)):
    os.mkdir(os.path.join('./predict/', args.aimdir))    


img_size = pd.read_csv('../Data2018/test_img_size.csv', header = None)
print(img_size.iloc[0])
       

net.eval()
with torch.no_grad():
    for i, (images, imgnam) in enumerate(testloader):
        imgnam = imgnam[0]
        
        if img_size.iloc[i][0] == imgnam:
            imgnam = imgnam.replace('.jpg', '_segmentation.png')
            w, h = img_size.iloc[i][1], img_size.iloc[i][2]
        else:
            print(imgnam, 'not found!')
            break
            
        images = images.to(device)#.squeeze()

        # Forward pass
        outputs = net(images)
        #print(outputs)
        outputs = F.softmax(outputs, dim=1)
#        predicted = outputs[0,1,:,:]

        _, predicted = outputs.max(1)

        predicted = 255*predicted.cpu().type(torch.ByteTensor).squeeze()#.numpy()
        img = transforms.ToPILImage(mode='L')(predicted)
        img = img.rotate(0)
        img = img.resize((w, h), Image.NEAREST) 

       
        img.save(os.path.join('./predict/', args.aimdir, imgnam))
        

    
