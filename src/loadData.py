#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import listdir,system
from os.path import isfile, join
import numpy as np
from sklearn import preprocessing


# In[6]:


class Data():
    def __init__(self):
        self.load = []
        self.load_data()
        self.load4D = []
        self.load4D_Y = []
        self.load_train4D()
        self.load6D = []
        self.load6D_Y = []
        self.load_train6D()
        
    def load_data(self):
        with open('case01.txt','r') as f :
            self.load.clear()
            for line in f.readlines():
                self.load.append(list(map(float,line.strip().split(','))))
            self.load = np.array(self.load)
        origin_point = self.load[0]
    
    def load_train4D(self):
        temp = []
        with open('data/train4dAll.txt','r') as f :
            self.load4D.clear()
            for line in f.readlines():
                temp = list(map(float,line.strip().split(' ')))
                self.load4D.append(temp[0:3])
                self.load4D_Y.append([temp[3]])
            self.load4D = np.array(self.load4D)
            self.load4D_Y = np.array(self.load4D_Y)
            print('len_load4D:', len(self.load4D))
            
    def load_parameters(self):
        temp = []
        temp_W = []
        temp_centers = []
        temp_beta = []
        with open('data/RBFN_params.txt','r') as f :
            for line in f.readlines():
                temp = list(map(float,line.strip().split(' ')))
                temp_W.append([temp[0]])
                temp_centers.append(temp[1:4])
                temp_beta.append([temp[4]])
            temp_W = np.array(temp_W)
            temp_centers = np.array(temp_centers)
            temp_beta = np.array(temp_beta)
        print('temp_W:', temp_W)
        print('temp_centers:', temp_centers)
        print('temp_beta:', temp_beta)
        return temp_W, temp_centers, temp_beta
    
    def getData(self):
        return self.load
    
    def getTrainData4d(self):
        return self.load4D, self.load4D_Y
    
    def getTrainData6d(self):
        return self.load6D, self.load6D_Y
    
    def normalize(self, data):
        #data_normalized = preprocessing.normalize(data, norm='l2')
        self.Min_Max_Scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        MinMax_Data = self.Min_Max_Scaler.fit_transform(data)
        return MinMax_Data
    
    def inverse_normalize(self, data):
        inverse_Data = self.Min_Max_Scaler.inverse_transform(data)
        return inverse_Data
    
    def normalize_Y(self, data):
        #data_normalized = preprocessing.normalize(data, norm='l2')
        self.Scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        MinMax_Data = self.Scaler.fit_transform(data)
        return MinMax_Data
    
    def inverse_normalize_Y(self, data):
        inverse_Data = self.Scaler.inverse_transform(data)
        return inverse_Data
    
    def normalize_input(self, data):
        data_normalized = preprocessing.normalize(data, norm='l2')
        return data_normalized


# In[7]:


'''
if __name__ == "__main__":
    data = Data()
'''


# In[ ]:





# In[ ]:





# In[ ]:




