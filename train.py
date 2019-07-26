#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 21:35:06 2019

@author: chenzj
"""

import sys

from datasets import DatasetTrain, DatasetVal # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
import lovasz_losses

sys.path.append("/home/chenzj/ISIC/segmentation/models")
from cenet import CE_Net_
from unet import UNet


sys.path.append("/home/chenzj/ISIC/segmentation/utils")
from utils import add_weight_decay, progress_bar

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

import os
import argparse


torch.manual_seed(1)           
torch.cuda.manual_seed(1)      


parser = argparse.ArgumentParser(description='Transferlearning...')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--max_epoch', default=30, type=int, help='max epoch to train')
args = parser.parse_args()





device = 'cuda' if torch.cuda.is_available() else 'cpu'
#best_acc = 0  # best test accuracy
best_iou = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data Prepare
print('==> Preparing data..')



transform_img = transforms.Compose([

    transforms.RandomChoice([
        transforms.ColorJitter(brightness=0.4),
        transforms.ColorJitter(contrast=0.4), 
        transforms.ColorJitter(saturation=0.3),
        transforms.ColorJitter(hue=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.15, hue=0.1), 
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3), 
        transforms.ColorJitter(brightness=0.4, contrast=0.33, saturation=0.2, hue=0.2), 
    ]),
    transforms.ToTensor(),

    transforms.Normalize(( 0.7082, 0.5715, 0.5273), 
                         ( 0.0981, 0.1134, 0.1276)),
]) 
    
    

transform_vali_img = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(( 0.7082, 0.5715, 0.5273), 
                         ( 0.0981, 0.1134, 0.1276)),
]) 



train_dataset = DatasetTrain(img_path='/home/chenzj/ISIC/Data2018/2018trainData/', label_path='/home/chenzj/ISIC/Data2018/2018trainlabel/', 
                             transform=None, transform_img=transform_img)

val_dataset = DatasetVal(img_path='/home/chenzj/ISIC/Data2018/2018validationData/', label_path='/home/chenzj/ISIC/Data2018/2018valiLabel/', 
                             transform=None, transform_img=transform_vali_img)

trainloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=8,
                                          shuffle=True,
                                          num_workers=2,
                                          drop_last= True)

testloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=8,
                                         shuffle=False,
                                         num_workers=2)



# Model
print('==> Building model..')

net = CE_Net_().to(device)
#net = CE_Net_backbone_DAC_with_inception().to(device)
#net = UNet(3,2).to(device)

#print(net)

if device == 'cuda':
    cudnn.benchmark = True





if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint_v'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint_v/ckptv1.t7')
    net.load_state_dict(checkpoint['net'])
    best_iou = checkpoint['IOU']
#    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
#    print(best_acc)
    print('best_iou:', best_iou)
    print('start_epoch:', start_epoch)
#    best_acc = 0
#    start_epoch = 0
    
'''
for param in net.parameters():
    param.requires_grad = False
'''

'''
if not args.resume:
    for m in net.modules():
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight, gain=1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight, gain=1)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
'''



# Loss and optimizer
#criterion = nn.CrossEntropyLoss()

#criterion = nn.BCELoss() 
#criterion =  focal_loss.FocalLoss(class_num = 2, alpha= w, gamma=2)
           

params = add_weight_decay(net, l2_value=0.0000001)
#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(params, lr=args.lr, eps=1e-08)
#optimizer = optim.ASGD(net.parameters(), lr=args.lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.max_epoch, eta_min=0.000000001, last_epoch=-1)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    lr_scheduler.step()

    net.train()
    train_loss = 0
    iou = 0
    
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
             
        #outputs = F.softmax(outputs, dim=1)
        loss = lovasz_losses.lovasz_softmax(outputs, targets)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        _, predicted = outputs.max(1)      
        iou_epoch = lovasz_losses.iou_binary(predicted, targets, ignore=255)   
        iou += iou_epoch        


        total += targets.size(0)*targets.size(1)*targets.size(2)
        correct += predicted.eq(targets).sum().item()
        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | IOU: %.3f | Acc: %.3f'
            % (train_loss/(batch_idx+1), iou/(batch_idx+1), 100.*correct/total ))

      

def test(epoch):
    global best_iou
    net.eval()
    test_loss = 0
    iou = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)           

        
            #outputs = F.softmax(outputs, dim=1)
            loss = lovasz_losses.lovasz_softmax(outputs, targets, ignore=255)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            
            iou_epoch = lovasz_losses.iou_binary(predicted, targets, ignore=255)   
            iou += iou_epoch            
            

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | IOU: %.3f'
                % (test_loss/(batch_idx+1), iou/(batch_idx+1) ))

    # Save checkpoint.
    iou = iou/len(testloader)

    if iou > best_iou:
        print('Saving..')
        state = {
            'net': net.state_dict(),
         #   'acc': acc,
            'IOU': iou,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint_v'):
            os.mkdir('checkpoint_v')
        torch.save(state, './checkpoint_v/ckptv1.t7')
        best_iou = iou


for epoch in range(start_epoch, start_epoch + args.max_epoch):
    train(epoch)
    test(epoch)
    
    
