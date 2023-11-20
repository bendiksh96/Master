import sys
sys.path.append(r'C:\Users\Lenovo\Documents\Master\Methods')
import numpy as np
import random as rd
from problem_func import *
from jde import *
from shade import *


class DEVO:
    def __init__(self, dim, problem_func, method):
        self.problem_func   = problem_func
        self.method         = method
        self.dim            = dim
        self.iter           = 0
        self.nfe            = 0
        
    #Initialize population(s)        
    def intialize_population(self, xmin, xmax, num_ind):
        self.xmin,self.xmax = xmin,xmax
        self.num_ind        = num_ind
        self.individual     = np.ones((self.num_ind, self.dim))
        self.likelihood     = np.ones((self.num_ind))
        
        #Initialize every individuals position and likelihood
        Data = Problem_Function(self.dim)
        for i in range(self.num_ind):
            for j in range(self.dim):
                self.individual[i,j] = np.random.uniform(xmin, xmax)
            self.likelihood[i] = Data.evaluate(self.individual[i,:], self.problem_func)
        self.nfe += self.num_ind
    
    
    def evolve(self, maxiter):
        if self.method == 'jde':
            mod = jDE(self.individual, self.likelihood, self.problem_func)
            while self.iter < maxiter:
                mod.evolve()
                self.check_oob()
                
                self.nfe  += self.num_ind
                self.iter +=1
        if self.method == 'shade':
            mod = SHADE(self.individual, self.likelihood, self.problem_func)
            while self.iter < maxiter:
                mod.evolve()
                self.check_oob()
                
                self.nfe  += self.num_ind
                self.iter +=1
        
        if self.method == 'ddms':
            pass
        
        if self.method == 'bat':
            pass
        
        if self.method == 'double_shade':
            pass
        
        
    #Sjekker rammebetingelser
    def check_oob(self):
        var = 0
        for i in range(self.num_ind):
            for j in range(self.dim):
                if  self.individual[i,j] < self.xmin:
                    var = self.xmin - (self.individual[i,j] - self.xmin)
                    if var > self.xmax:
                        self.individual[i,j] = 0
                        print('Some error has occured in oob')
                    else:
                        self.individual[i,j] = var

                if  self.individual[i,j] > self.xmax:
                    var  = self.xmax - (self.individual[i,j] - self.xmax)
                    if var < self.xmin:
                        self.individual[i,j] = 0
                        print('Some error has occured in oob')
                    else:
                        self.individual[i,j] = var

        
        
a = DEVO(2, 'Eggholder', 'jde')
a.intialize_population(-512,512,100)
a.evolve(50)
print(a.individual)