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


class ListGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, file_list, example_count=270, batch_size=32, mean=0, disper=1, debug=False):
        'Initialization'
        self.file_list = file_list
        # number of examples per person
        self.example_count = example_count
        self.batch_size = batch_size
        self.mean = mean
        self.disper = disper
        self.debug = debug
        
    def __getitem__(self, index):
        'Generate one batch of data'
        rand0 = random.randint(0,len(self.file_list)-1)
        
        p_ind = re.compile('/([0-9]+)')
        file1 = self.file_list[rand0]
        ind = p_ind.search(file1).group(1)

        same_file_list = [file for file in self.file_list if (p_ind.search(file).group(1) == ind)]
        same_file_list.remove(file1)
        x1 = pd.read_csv(file1, delimiter=' ', header=None).drop([0,1,402],axis=1)
        x1 = (x1 - self.mean)/self.disper
        l = []
        
        for i in range(int(self.batch_size/2)):
            name1 = ''
            rand1 = random.randint(0,len(same_file_list)-1)
            name1 = same_file_list[rand1]
            x_same = pd.read_csv(name1 , delimiter=' ', header=None).drop([0,1,402],axis=1)
            x_same = (x_same - self.mean)/self.disper
            x_same['label'] = 1
            l.append(x_same)
            
            ind2 = ind
            while ind2 == ind:
                rand2 = random.randint(0,len(self.file_list)-1)
                name2 = self.file_list[rand2]
                ind2 = p_ind.search(name2).group(1)
                
            x_another = pd.read_csv(name2, delimiter=' ', header=None).drop([0,1,402],axis=1)
            x_another = (x_another - self.mean)/self.disper
            x_another['label'] = 0
            l.append(x_another)
        partners = pd.concat(l, axis=0) 
        x2 = np.tile(x1.values, (int(self.batch_size/2)*2, 1))
        y = partners.pop('label')
        if (self.debug):
            print('First file = %s' % file1)
            print('Same file = %s' % name1)
            print('Another file = %s' % name2)
        return [x2, partners.values], y.values
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.file_list) / self.batch_size)
    
    def on_epoch_end(self):
        pass

class ListGenerator2(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, file_list, example_count=270, batch_size=32, mean=0, disper=1, debug=False):
        'Initialization'
        self.file_list = file_list
        # number of examples per person
        self.example_count = example_count
        self.batch_size = batch_size
        self.mean = mean
        self.disper = disper
        self.debug = debug
        
    def __getitem__(self, index):
        'Generate one batch of data'
        p_ind = re.compile('/([0-9]+)')
        l0 = []
        l1 = []
        for i in range(int(self.batch_size/2)):
            rand0 = random.randint(0,len(self.file_list)-1)
            file0 = self.file_list[rand0]
            ind = p_ind.search(file0).group(1)
            same_file_list = [file for file in self.file_list if (p_ind.search(file).group(1) == ind)]
            same_file_list.remove(file0)
            x0 = pd.read_csv(file0, delimiter=' ', header=None).drop([0,1,402],axis=1)
            x0 = (x0 - self.mean)/self.disper
            l0.append(x0)
            l0.append(x0)
        
            name1 = ''
            rand1 = random.randint(0,len(same_file_list)-1)
            name1 = same_file_list[rand1]
            x_same = pd.read_csv(name1 , delimiter=' ', header=None).drop([0,1,402],axis=1)
            x_same = (x_same - self.mean)/self.disper
            x_same['label'] = 1
            l1.append(x_same)
            
            ind2 = ind
            while ind2 == ind:
                rand2 = random.randint(0,len(self.file_list)-1)
                name2 = self.file_list[rand2]
                ind2 = p_ind.search(name2).group(1)
                
            x_another = pd.read_csv(name2, delimiter=' ', header=None).drop([0,1,402],axis=1)
            x_another = (x_another - self.mean)/self.disper
            x_another['label'] = 0
            l1.append(x_another)
        prince = pd.concat(l0, axis=0)
        partners = pd.concat(l1, axis=0) 
        y = partners.pop('label')
        if (self.debug):
            print('First file = %s' % file0)
            print('Same file = %s' % name1)
            print('Another file = %s' % name2)
        return [prince.values, partners.values], y.values
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.file_list) / self.batch_size)
    
    def on_epoch_end(self):
        pass
