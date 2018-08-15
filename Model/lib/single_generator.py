#coding=utf-8
    
import numpy as np
import pandas as pd
import keras
import random
import os.path
import re
import glob

def name(ind, num1):
    if isinstance(num1, int):
        a = num1 % 30 +1
        b = num1 // 30 +1
        return './../Data/train_db/%s_%02d%03d.txt' % (ind, b, a)
    return './../Data/train_db/%s_%s.txt' % (ind, num1)


class SingleGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, file_list, id_list, num_classes=194, batch_size=32, mean=0, disper=1):
        'Initialization'
        # id_list is list of person ids
        self.num_classes = num_classes
        self.file_list = file_list
        self.id_list = sorted(set(map(int, id_list)))
        self.batch_size = batch_size
        self.mean = mean
        self.disper = disper
        
    def __getitem__(self, index):
        'Generate one batch of data'
        p = re.compile('/([0-9]+)')
        l = []
        for i in range(self.batch_size):
            rand = random.randint(0,len(self.file_list)-1)
            file = self.file_list[rand]
            x = pd.read_csv(file , delimiter=' ', header=None).drop([0,1,402],axis=1)
            x = (x - self.mean)/self.disper
            ind = p.search(file).group(1) 
            x['label'] = self.id_list.index(int(ind))
            l.append(x)
            
        partners = pd.concat(l, axis=0) 
        
        X = partners
        y = X.pop('label')
        y = keras.utils.to_categorical(y, self.num_classes)
        return X.values, y
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.file_list)/self.batch_size)
    
    def on_epoch_end(self):
        pass