# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 23:19:58 2020

@author: Dimo
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os

import time
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

from random import shuffle

from DS import LoadData
from resnet18 import Resnet18
from loss import OhemCELoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ModelManager:
    def __init__(self,ModelRoot):
        self.ModelRoot = ModelRoot
        self.model = Resnet18()
        self.model = self.model.to(device)
        dataSource = LoadData(root='./data')

        a,b,self.test_loader = dataSource()
        self.known = a
        self.unknown = b
        self.KnownLoader = torch.utils.data.DataLoader(
                            dataset=a,
                            batch_size=100,
                            shuffle=True)
        self.UnknownLoader = torch.utils.data.DataLoader(
                              dataset=b,
                              batch_size=100,
                              shuffle=True)
        # self.loss = OhemCELoss(thresh=0.5, n_min=28*28, ignore_lb=255)

        
        
        
        
        if os.path.exists(self.ModelRoot+'/resnet18.pth'):
            self.model.load_state_dict(torch.load(self.ModelRoot+'/resnet18.pth'))
            print('pretrained weights loaded!')
        elif not os.path.exists(self.ModelRoot):
            os.mkdir(ModelRoot)
            
        
            
    def train_known(self,n_epochs,ite,max_ite):
        self.model.train()
        avg_loss = []
        
        optim = torch.optim.Adam(self.model.parameters(), 
                                          lr=0.001, 
                                          weight_decay=0)
        # lossOH = OhemCELoss(0.2,50) #cause batch is 100
        looss = nn.CrossEntropyLoss()
        print('Init known training...')
        number = 0
        for epoch in range(n_epochs):
            for i, batch in enumerate(self.KnownLoader):
                lb = batch[1].to(device)
                print(len(lb))
                img = batch[0].to(device)
                # Training
                optim.zero_grad()
                # a,b,c,out = self.model(img)
                out = self.model(img)

                # print('outISS',out)
                loss = looss(out,lb)
                avg_loss = torch.mean(loss)
               
                loss.backward()
                optim.step()
                print(avg_loss)
                number+=1

                if number%10 == 0:
                  print(number)
                print("\n---- Epoch: [{0}/{1}] batch: [{2}/{3}] iteration: [{4}/{5}] ----\t".
                      format((epoch+1), n_epochs, (i+1), len(self.KnownLoader), (ite+1), max_ite))
        # Save checkpoint
        torch.save(self.model.state_dict(),"./data/resnet18.pth")
        number =0

    def train_known(self,n_epochs,ite,max_ite):
        self.model.train()
        avg_loss = []
        
        optim = torch.optim.Adam(self.model.parameters(), 
                                          lr=0.001, 
                                          weight_decay=0)
        # lossOH = OhemCELoss(0.2,50) #cause batch is 100
        looss = nn.CrossEntropyLoss()
        print('Init known training...')
        number = 0
        for epoch in range(n_epochs):
            for i, batch in enumerate(self.KnownLoader):
                lb = batch[1].to(device)
                print(len(lb))
                img = batch[0].to(device)
                # Training
                optim.zero_grad()
                # a,b,c,out = self.model(img)
                out = self.model(img)

                # print('outISS',out)
                loss = looss(out,lb)
                avg_loss = torch.mean(loss)
               
                loss.backward()
                optim.step()
                print(avg_loss)
                number+=1

                if number%10 == 0:
                  print(number)
                print("\n---- Epoch: [{0}/{1}] batch: [{2}/{3}] iteration: [{4}/{5}] ----\t".
                      format((epoch+1), n_epochs, (i+1), len(self.KnownLoader), (ite+1), max_ite))

    def train_Unnown(self,dataindex,n_epochs,ite,max_ite):
        # print(self.unknown[dataindex])
        dataset = self.unknown[dataindex.astype(int)]
        datasetLoader =torch.utils.data.DataLoader(
                            dataset=dataset,
                            batch_size=100,
                            shuffle=True)
        self.model.train()
        avg_loss = []
        
        optim = torch.optim.Adam(self.model.parameters(), 
                                          lr=0.001, 
                                          weight_decay=0)
        # lossOH = OhemCELoss(0.2,50) #cause batch is 100
        looss = nn.CrossEntropyLoss()
        print('Init Unknown training...')
        number = 0
        for epoch in range(n_epochs):
            for i, batch in enumerate(datasetLoader):
                lb = batch[1].to(device)
                print(len(lb))
                img = batch[0].to(device)
                # Training
                optim.zero_grad()
                # a,b,c,out = self.model(img)
                out = self.model(img)

                # print('outISS',out)
                loss = looss(out,lb)
                avg_loss = torch.mean(loss)
               
                loss.backward()
                optim.step()
                print(avg_loss)
                number+=1

                if number%10 == 0:
                  print(number)
                print("\n---- Epoch: [{0}/{1}] batch: [{2}/{3}] iteration: [{4}/{5}] ----\t".
                      format((epoch+1), n_epochs, (i+1), len(dataset), (ite+1), max_ite))
    
    def _compute_scores(self,y_true, y_pred):

            folder = "test"

            labels = list(range(10)) # 4 is the number of classes: {0,1,2,3}
            confusion = confusion_matrix(y_true, y_pred, labels=labels)
            precision, recall, fscore, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='macro')
            accuracy = accuracy_score(y_true, y_pred)

            # print(confusion)

            scores = {}
            scores["{}/accuracy".format(folder)] = accuracy
            scores["{}/precision".format(folder)] = precision
            scores["{}/recall".format(folder)] = recall
            scores["{}/f1".format(folder)] = fscore

            precision, recall, fscore, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, labels=labels, average=None)

            for i in range(len(labels)):
                prefix = "{}_{}/".format(folder, i)
                scores[prefix + "precision"] = precision[i]
                scores[prefix + "recall"] = recall[i]
                scores[prefix + "f1"] = fscore[i]

            return scores

    def move_unknown(self,indx):
        self.known.append(self.unknown[indx])
        self.unknown.pop(indx)
        print('moved all new files')

    def predict_probability(self,budget):
        self.model.eval()
        avg_loss = []
        budgetE = budget
    
        looss = nn.CrossEntropyLoss()
        print('Init probability sampling method...')
        number = 0
        # Initialize the prediction and label lists(tensors)
        predlist=torch.zeros(0,dtype=torch.long, device='cpu')
        lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
        scores = []
        b = self.unknown
        probs = torch.zeros([len(b), 10])
        probs = torch.zeros([budget+1, 10])

        
        unknownLoader = torch.utils.data.DataLoader(
                              dataset=b,
                              batch_size=1,
                              shuffle=False)
        indexes = []
        
        with torch.no_grad():
          
                for i, batch in enumerate(unknownLoader):
                    lb = batch[1].to(device)
                    img = batch[0].to(device)
  
                    out = self.model(img)
                    prob = F.softmax(out, dim=1)
                    probs[i] = prob.cpu()
                    budgetE-=1
                    if(budgetE<0):break
                    indexes.append(i)
                    
        return indexes,probs
  


    def validate(self):
        self.model.eval()
        avg_loss = []
    
        looss = nn.CrossEntropyLoss()
        print('Init validate method...')
        number = 0
        # Initialize the prediction and label lists(tensors)
        predlist=torch.zeros(0,dtype=torch.long, device='cpu')
        lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
        scores = []

        with torch.no_grad():
                for i, batch in enumerate(self.test_loader):
                    lb = batch[1].to(device)
                    # print(len(lb),lb[0])
                    img = batch[0].to(device)
                    # Training
                    # optim.zero_grad()
                    # a,b,c,out = self.model(img)
                    out = self.model(img)
                    _, preds = torch.max(out, 1)

                    # Append batch prediction results
                    predlist=torch.cat([predlist,preds.view(-1).cpu()])
                    lbllist=torch.cat([lbllist,lb.view(-1).cpu()])
                    # print(preds.cpu().numpy(),lb.cpu().numpy())

                    scores = self._compute_scores(preds.view(-1).cpu(),lb.view(-1).cpu())

                    # print('outISS',out)
                    loss = looss(out,lb)
                    avg_loss = torch.mean(loss)
                
                    # loss.backward()
                    # optim.step()
                    
                    number+=1

                    if number%10 == 0:
                        print(number)
                        print('Average loss is : --- ',avg_loss)

        # Confusion matrix
        conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
        # print(conf_mat)

        # Per-class accuracy
        class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
        print(class_accuracy)       
        # Save checkpoint
        print("- accuracy: {:.3f}".format(scores["test/accuracy"]))
        print("- precision: {:.3f}".format(scores["test/precision"]))
        print("- recall: {:.3f}".format(scores["test/recall"]))
        print("- f1: {:.3f}".format(scores["test/f1"]))
        print("- classification_loss: {:.3f}".format(avg_loss))
        number =0
                
                
            