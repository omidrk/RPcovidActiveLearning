# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 20:52:09 2020

@author: Dimo
"""
import torch
import numpy as np

from resnet18 import Resnet18
from DS import LoadData
from ModelManager import ModelManager

if __name__ == "__main__":
    manager = ModelManager(ModelRoot='./model')

    #train for the known model
    manager.train_known(1,30,40)
    manager.validate()

    #start unknown part
    probs = manager.predict_probability(500) #budget of searching
    # print(len(temp[1]))
    logprob = torch.log(probs[1])
    E = logprob*probs[1]
    E = E.sum(1)
    probs1 = list(zip(probs[0],E))
    print(probs1)
    res = sorted(probs1, key = lambda x: x[1])
    indx = np.array(res[:100],dtype=np.int) #untill batch index
    indx = indx[:,0].astype('int32')
    print(indx)
    #train unknown
    manager.train_Unnown(indx,1,30,40)
    #Move Items from unknown to known
    manager.move_unknown(indx)

    # len(E)
    # print(E)
    # temp2 = list([temp[0],E])