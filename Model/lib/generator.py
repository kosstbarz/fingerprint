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
    
def cros(x1, x2):
    x1['key'] = 1
    x2['key'] = 1
    df = pd.merge(x1, x2, on='key')
    del df['key']
    return df


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, id_list, example_count=270, batch_size=32, mean=0, disper=1):
        'Initialization'
        # id_list is list of person ids
        self.id_list = id_list
        # number of examples per person
        self.example_count = example_count
        self.batch_size = batch_size
        self.mean = mean
        self.disper = disper
        
    def __getitem__(self, index):
        'Generate one batch of data'
        num = index % len(self.id_list)
        ind = self.id_list[num]
        p = re.compile('/([0-9]+)_([0-9]+)')
        num_list = [p.search(file).group(2) for file in glob.glob('./../Data/train_db/%s*' % ind)]
        x1 = pd.read_csv(name(ind, num_list[0]), delimiter=' ', header=None).drop([0,1,402],axis=1)
        x1 = (x1 - self.mean)/self.disper
        l = []
        for i in range(int(self.batch_size/2)):
            name1 = ''
            rand1 = random.randint(1,len(num_list)-1)
            name1 = name(ind,num_list[rand1])
            x_same = pd.read_csv(name1 , delimiter=' ', header=None).drop([0,1,402],axis=1)
            x_same = (x_same - self.mean)/self.disper
            x_same['label'] = 1
            l.append(x_same)
            
            num2=num
            name2 = ''
            while num2 == num or not os.path.isfile(name2):
                rand2 = random.randint(0,self.example_count-1)
                rand3 = random.randint(0,len(self.id_list)-1)
                num2 = rand3
                ind2 = self.id_list[num2]
                name2 = name(ind2,rand2)
            x_another = pd.read_csv(name2, delimiter=' ', header=None).drop([0,1,402],axis=1)
            x_another = (x_another - self.mean)/self.disper
            x_another['label'] = 0
            l.append(x_another)
        partners = pd.concat(l, axis=0) 
        
        X = cros(x1, partners)  
        y = X.pop('label')
        return X.values, y.values
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.id_list)
    
    def on_epoch_end(self):
        pass
