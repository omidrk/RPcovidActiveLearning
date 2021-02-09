# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 20:56:43 2020

@author: Dimo
"""

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from PIL import Image


## load mnist dataset
use_cuda = torch.cuda.is_available()

class LoadData:
    def __init__(self,root,mode='train'):        
        self.root = root # example = './data'
        self.mode = mode
        if not os.path.exists(root):
            os.mkdir(root)
    
    def __call__(self):
        
    
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (1.0,)),
                                    transforms.Resize(64)])
                                    #transforms.Resize((64,64),interpolation=Image.BILINEAR)
        # if not exist, download mnist dataset
        train_set = dset.MNIST(root=self.root, train=True, transform=trans, download=True)
        test_set = dset.MNIST(root=self.root, train=False, transform=trans, download=True)
        
        #split for known and unknown
        fullLen = len(train_set)
        a,b = data.random_split(train_set, [8000, 52000],
                                generator=torch.Generator().manual_seed(42))
        

        batch_size = 30

        train_known_loader = torch.utils.data.DataLoader(
                         dataset=a,
                         batch_size=batch_size,
                         shuffle=True)
        
        train_Unknown_loader = torch.utils.data.DataLoader(
                         dataset=b,
                         batch_size=batch_size,
                         shuffle=True)
        
        test_loader = torch.utils.data.DataLoader(
                        dataset=test_set,
                        batch_size=batch_size,
                        shuffle=False)
        
        print('total trainning known batch number: {}'.format(len(a)))
        print('total trainning unknown batch number: {}'.format(len(b)))
        print('total testing batch number: {}'.format(len(test_loader)))
        
        # return train_known_loader,train_Unknown_loader,test_loader
        return a,b,test_loader
    
if __name__ == "__main__":
    myclass = LoadData(root='./data')
    train_known_loader,train_Unknown_loader,test_loader = myclass()