#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np  
import random   
import matplotlib.pyplot as plt
import importer
from loadData import Data
from RBFN import RBF
from scipy.linalg import norm, pinv
 
class pso:  
    def __init__(self,pN,max_iter, learn1, learn2,input_dim, input_x, output_y, weight, centers, beta):
        self.len_weight = len(weight)
        self.weight = weight
        self.centers = centers
        self.num_centers = len(centers) * input_dim
        self.beta = beta
        self.len_beta = len(beta) * input_dim
        self.input_x = input_x
        self.input_dim = input_dim
        self.output_y = output_y
        self.theta = np.random.uniform(-1,1)
        self.w = 0.8    
        self.learn_1 = learn1     
        self.learn_2 = learn2
        self.pN = pN                
        self.dim = self.cal_dim(self.theta, weight, centers, beta) 
        self.max_iter = max_iter   
        self.X = np.zeros((self.pN,self.dim))       
        print('self.X:', self.X)
        self.V = np.zeros((self.pN,self.dim))
        print('self.V:', self.V)
        self.pbest = np.zeros((self.pN,self.dim))  
        self.gbest = np.zeros((1,self.dim))
        self.p_fit = np.zeros(self.pN)               
        self.fit = 1e10           
        #self.print_data()
        
    def print_data(self):
        print('self.input_x:', self.input_x)
        print('self.weight', self.weight)
        print('self.centers: ',self.centers)
        print('self.beta:', self.beta)
        
    def cal_dim(self, theta, w, centers, beta):
        dim = []
        print(theta)
        print(w)
        print(centers)
        print(beta)
        dim = np.hstack((dim,theta))
        for k in range(0, len(w)):
            dim = np.hstack((dim, w[k]))
        for i in range(0, len(centers)):
            dim = np.hstack((dim, centers[i]))
        for j in range(0, len(beta)):
            dim = np.hstack((dim, beta[j]))
        return len(dim)
    
    def basicFunc(self, c, d, beta):
        return np.exp(beta * norm(c - d) ** 2)
    
    def _calcAct(self):
        G = np.zeros((self.input_x.shape[0], len(self.centers)), dtype = np.float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(self.input_x):
                G[xi, ci] = self.basicFunc(c, x, -1/2*(self.beta[ci][0]**2))
        return G
    
    def get_fitness(self,pred):
        return pred + 1e-3 - np.min(pred)
    
    def E(self, y, f):
        total = 0
        for i in range(0, len(y)):
            total += (y[i] - f[i])**2
        total = total/2
        return total
    
    def function(self, x):
        self.set_parameter(x)
        rnd_idx = np.random.permutation(self.input_x.shape[0])[:len(self.centers)]
        self.centers = [self.input_x[i, :] for i in rnd_idx]
        G = self._calcAct()
        self.weight = np.dot(pinv(G), self.output_y)
        f = np.dot(G, self.weight)
        sum = self.E(self.output_y,f)
        return sum
    
    def set_parameter(self, x):
        self.theta = x[0]
        self.weight = x[1:len(self.weight)+1]
        temp_w = []
        for i in range(0, len(self.weight)):
            temp_w.append([self.weight[i]])
        self.weight = temp_w
        self.centers = x[1+len(self.weight):(1+len(self.weight)+self.num_centers)]
        self.centers = np.split(self.centers, self.num_centers/self.input_dim, axis=0)
        self.beta = x[1+len(self.weight)+self.num_centers:]
        self.beta = np.split(self.beta, self.len_beta/self.input_dim, axis=0)
  
    def initialize(self):  
        for i in range(self.pN):  
            for j in range(self.dim):  
                self.X[i][j] = random.uniform(0,1)  
                self.V[i][j] = random.uniform(0,1)  
            self.pbest[i] = self.X[i]  
            tmp = self.function(self.X[i])  
            self.p_fit[i] = tmp  
            if(tmp < self.fit):  
                self.fit = tmp  
                self.gbest = self.X[i]  

    def training(self):  
        Fit = []  
        for t in range(self.max_iter):  
            for i in range(self.pN):         
                temp = self.function(self.X[i])  
                if(temp < self.p_fit[i]):        
                    self.p_fit[i] = temp  
                    self.pbest[i] = self.X[i]  
                if(self.p_fit[i] < self.fit): 
                    self.gbest = self.X[i]  
                    self.fit = self.p_fit[i]  
            for i in range(self.pN):  
                self.V[i] = self.w*self.V[i] + self.learn_1*(self.pbest[i] - self.X[i]) +                             self.learn_2*(self.gbest - self.X[i])  
                self.X[i] = self.X[i] + self.V[i]  
            Fit.append(self.fit)  
            print('self.fit: ', self.fit)
            print('self.gbest:', self.gbest)
        return Fit 
    
    def get_parameter(self):
        self.set_parameter(self.gbest)
        return self.theta, self.weight, self.centers, self.beta


# In[2]:


if __name__ == "__main__":
    #----------------------程序執行-----------------------
    D = Data()
    x, y = D.getTrainData4d()
    x = D.normalize(x)
    y = D.normalize(y)
    print('lenY', len(y))
    print('y:', y)
    rbf = RBF(3, 10, 1)
    x_dim, w, centers, beta = rbf.get_parameter()
    print('w:', w)
    print('centers:', centers)
    print('beta:', beta)
    my_pso = pso(5, 10,x_dim, x, y, w, centers, beta)  
    my_pso.init_Population()  
    fitness = my_pso.iterator()
    print('fitness: ', fitness)
    theta, w, centers, beta = my_pso.get_parameter()
    print('w____:', w)
    print('centers_____:', centers)
    print('beta_____:', beta)
    rbf.set_parameter(theta, w, centers, beta)
    rbf.train(x,y)
    print('success!!!')


# In[ ]:




