# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:06:02 2024

@author: Bendik Selvaag-Hagen
"""
import sys
sys.path.append(r'C:\Users\Bendik Selvaag-Hagen\OneDrive - Universitetet i Oslo\Documents\GitHub\Master')
import numpy as np
import matplotlib.pyplot as plt


class Super_Data:
    def __init__(self, dim, problem_func, log_thresh):
        self.problem_func   = problem_func
        self.dim            = dim
        self.hist_data      = []
        self.nfe            = 0
        self.best           = 10
        self.log_thresh     = log_thresh
        self.Data    = Problem_Function(self.dim)
        self.Data.param_change(0, self.log_thresh)        
        #self.open_data()
        
    def initialize_population(self, xmin, xmax, num_ind):
        self.xmin,self.xmax     = xmin,xmax
        self.num_ind            = num_ind
        self.individual         = np.ones((self.num_ind, self.dim))
        self.likelihood         = np.ones((self.num_ind))
        self.actual_likelihood  = np.ones_like(self.likelihood)
        self.conv_iter          = 0

        for i in range(self.num_ind):
            var     = []
            for j in range(self.dim):
                self.individual[i,j] = np.random.uniform(xmin, xmax)
                var.append(self.individual[i,j])
            temp, z = self.Data.evaluate(self.individual[i,:], self.problem_func)
            self.likelihood[i] = temp
            var.append(z)
            self.hist_data.append(var)
        self.nfe += self.num_ind



    def evolve(self, max_nfe):
        self.v          = np.zeros_like(self.individual)
        sort            = np.argsort(self.likelihood, axis = 0)
        F = 0.1
        #self.abs_best   = self.likelihood[best_indexes[0]]
        #self.best_ind   = self.individual[best_indexes][0]
        
        for _ in range(max_nfe):
            for i in range(self.num_ind):
                ri1 = np.random.randint(0,self.num_ind)
                ri2 = np.random.randint(0,self.num_ind)
                ri3 = np.random.randint(0,self.num_ind)
                
                #rand/1/bin
                self.v[i] = self.individual[ri1] + F*(self.individual[ri2] - self.individual[ri3] )
    
            for i in range(self.num_ind):
                perceived, true = self.Data.evaluate(self.v[i], self.problem_func)
                self.nfe += 1
                #self.hist_data.append(self.v[i], true)
                if perceived < self.likelihood[i]:
                    u =  np.random.uniform(0,1)
                    if np.random.uniform(0,1)  < 0.8:
                        self.individual[i] = self.v[i]
                        self.likelihood[i] = perceived
                    elif u  < 0.4:
                        self.individual[i] = self.v[i]
                        self.likelihood[i] = perceived
                        
            if self.nfe > max_nfe:
                break
    def plot(self):
        
        plt.scatter(self.individual[:,0], self.individual[:,1])#, self.likelihood)
        plt.colorbar()
        plt.show()

dim = 3
problem_func = 'mod_Himmelblau'
log_thresh = 3.09
data_collector = Super_Data(dim, problem_func, log_thresh)

xmin, xmax = -5,5
num_ind = 1000

data_collector.initialize_population(xmin, xmax, num_ind)
data_collector.evolve(100000)
data_collector.plot()