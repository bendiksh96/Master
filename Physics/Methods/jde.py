import numpy as np
import sys 
sys.path.append(r"C:\Users\Lenovo\Documents\GitHub\Master\Physics")
from physical_eval import *
class jDE:
    def __init__(self, num_ind):
        self.num_ind        = num_ind
        self.dim            = 3
        self.Flist          = [0.1 for p in range(self.num_ind)]
        self.CRlist         = [0.1 for p in range(self.num_ind)]
        self.tau1,self.tau2 = 0.1,0.1
        
        self.xs             = XSection()
        self.nfe            = 0
        self.hist_data      = []
    
    def initialize_population(self):
        #gluino, neutralino, squark
        self.ind_ind    = ['g', 'n' , 'q']
        self.individual = np.zeros((self.num_ind, self.dim))
        self.likelihood = np.zeros(self.num_ind)
        self.v          = np.zeros_like(self.individual)
        self.xmin_arr = [500 ,  0 , 500]
        self.xmax_arr = [3000,2000,3000]
        
        #Initialize 
        for p in range(self.num_ind):
            for j in range(self.dim):
                if self.ind_ind[j] == 'g' or self.ind_ind[j] == 'q':
                    self.individual[p, j] = np.random.uniform(500,3000)
                if self.ind_ind[j] == 'n':
                    self.individual[p, j] = np.random.uniform(0  ,2000)
            # print(self.individual[p])
            temp, true_likelihood, signal, section = self.eval_likelihood_ind(self.individual[p])
            k = [self.individual[p,j] for j in range(self.dim)] + [true_likelihood] + [signal] + [section]
            self.hist_data.append(k)    
            self.likelihood[p] = temp
        self.nfe = self.num_ind
        
    def evolve(self):
        sort_index      = np.argsort(self.likelihood, axis = 0)
        best_index      = sort_index[0]
        self.abs_best   = self.likelihood[best_index]
        for i in range(self.num_ind):
            ri1 = np.random.randint(self.num_ind)
            ri2 = np.random.randint(self.num_ind)
            ri3 = np.random.randint(self.num_ind)
            
            #rand/1 scheme
            self.v[i] = self.individual[ri1] + self.Flist[i] * (self.individual[ri2] - self.individual[ri3])
        #Muter        
        for i in range(self.num_ind):
            randint = np.random.uniform(0,1)
            if randint < self.CRlist[i]:
                self.v[i], candidate_status = self.check_oob(self.v[i])
                if candidate_status:
                    # print(self.v[i])
                    temp, true_likelihood, signal, section = self.eval_likelihood_ind(self.individual[i])
                    k = [self.individual[i,j] for j in range(self.dim)] + [true_likelihood] + [signal] + [section]
                    self.hist_data.append(k)
                    self.nfe += 1
                    if temp < self.likelihood[i]:
                        self.individual[i] = self.v[i]
                        self.likelihood[i] = temp

        #Crossover
        for i in range(self.num_ind):
            ru1 = np.random.uniform(0,1)
            ru2 = np.random.uniform(0,1)
            ru3 = np.random.uniform(0,1)
            ru4 = np.random.uniform(0,1)
            if ru1 < self.tau1:
                self.Flist[i] += ru2 * self.Flist[i]
            if ru3 < self.tau2:
                self.CRlist[i] = ru4


    def check_oob(self, candidate):
        candidate_status = True
        for j in range(self.dim):
            xmin, xmax = self.xmin_arr[j], self.xmax_arr[j]
            if candidate[j] < xmin:
                var = xmin - (candidate[j] - xmin)
                if var > xmax or var < xmin:
                    candidate_status = False
                else:
                    candidate[j] = var
                    
            if  candidate[j] > xmax:
                var  = xmax - (candidate[j] - xmax)
                if var < xmin or var > xmax:
                    candidate_status = False
                else:
                    candidate[j] = var
        return candidate, candidate_status
    
    
    def eval_likelihood_ind(self, individual):
        percieved_val, true_val, signal, section = self.xs.evaluate(individual)
        return percieved_val, true_val, signal, section
