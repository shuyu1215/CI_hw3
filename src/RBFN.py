#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.linalg import norm, pinv
import importer
from loadData import Data

np.random.seed(20)

class RBF:
    def __init__(self, input_dim, num_centers, out_dim):
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.out_dim = out_dim
        self.beta = [np.random.uniform(-1,1, input_dim) for i in range(num_centers)]
        self.centers = [np.random.uniform(-1,1, input_dim) for i in range(num_centers)]
        self.W = np.random.random((self.num_centers, self.out_dim))
    
    def basicFunc(self, c, d, beta):
        return np.exp(beta * norm(c - d) ** 2)
    
    def _calcAct(self, X):
        G = np.zeros((X.shape[0], self.num_centers), dtype = np.float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self.basicFunc(c, x, -1/2*(self.beta[ci][0]**2))
        return G
    
    def train(self, X, Y):
        rnd_idx = np.random.permutation(X.shape[0])[:self.num_centers]
        self.centers = [X[i, :] for i in rnd_idx]
        G = self._calcAct(X)
        self.W = np.dot(pinv(G), Y)
        
    def predict(self, X, W):
        G = self._calcAct(X)
        Y = np.dot(G, W)
        return Y
    
    def get_parameter(self):
        return self.input_dim, self.W, self.centers, self.beta
    
    def set_parameter(self, theta, w, centers, beta):
        self.theta = theta
        self.W = w
        self.centers = centers
        self.beta = beta
        
    def get_weight(self):
        return self.W
    
if __name__ == "__main__":
    data4D = open("/Users/chengshu-yu/Documents/train/train4D.txt",'w+')
    D = Data()
    n = 100
    #x = np.linspace(-1, 1, n).reshape(n, 1)
    #y = np.sin(3 * (x+0.5)**3 - 1)
    #training
    x, y = D.getTrainData()
    x = D.normalize(x)
    y = D.normalize(y)
    #print('x:', x)
    #print('lenY', len(y))
    #print('y:', y)
    rbf = RBF(3, 200, 1)
    rbf.train(x,y)
    z = rbf.predict(x)
    y = D.inverse_normalize(y)
    z = D.inverse_normalize(z)
    for i in range(0, len(z)-1):
        print(str(y[i]), str(z[i]),file=data4D)
    data4D.close()
    plt.plot(y, 'k-', label = u'actually')
    plt.plot(z, 'r-', linewidth=2, label = u'predict')
    
    plt.xlim(-1.2, 1.2)
    plt.title(u'RBF', fontsize = 20, color = 'r')
    plt.legend(loc = 'upper left')
    plt.show()


# In[ ]:





# In[ ]:




