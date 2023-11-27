import numpy as np
import matplotlib.pyplot as plt 
import sys
# sys.path.append(r'C:\Users\Lenovo\Documents\Master')
from problem_func import *

class Bat:
    def __init__(self, individual, likelihood, problem_func, standard = True):
        self.individual = individual
        self.likelihood = likelihood
        self.dim        = len(individual[0])        
        self.num_ind    = len(likelihood)
        self.prob_func  = problem_func

        self.iter       = 0        
        self.BM         = [] 
        self.BM_val     = []
        self.velocity   = []
        self.pulse      = []

        
        if standard == True:
            self.rj          = .2
            self.rj0         = .2
            self.A           = .5
            self.eps         = .01
            self.gamma       = .9
            self.alfa        = .9
        self.init_vel()
    
    def set_limits(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
    
    def change_var(self, rj, rj0, A, eps, gamma, alfa):
        self.rj          = rj
        self.rj0         = rj0
        self.A           = A
        self.eps         = eps
        self.gamma       = gamma
        self.alfa        = alfa

    def evolve(self):
        sort = np.argsort(self.likelihood)
        best_ind = self.individual[sort][0]
        best_val = self.likelihood[sort][0]
        worst_val = self.likelihood[sort][-1]
        #Forslag 7D
        # super_val = -5216
        # best_profile = super_val + (pop+1)/2*(super_val - best_val)
        
        for i in range(self.num_ind):
            for j in range(self.dim):
                u = np.random.uniform(0,1)
                #Frequency
                f_j = best_val + (best_val-worst_val) * np.random.uniform(0,1)
                #self.velocity
                # self.velocity[i,j] = self.velocity[i,j] + 1/(100*pop+self.num_ind)*(self.individual[i,j] - best_ind[0][j]) * f_j[0][0]

                self.velocity[i,j] = self.velocity[i,j] +(self.individual[i,j] - best_ind[j]) * f_j
                #Update self.individual if
                if u < self.pulse[i,j]:
                    self.individual[i,j] = best_ind[j] + self.eps*A
                else:
                    self.individual[i,j] = self.velocity[i,j] + self.individual[i,j] 
                
                
                #oob
                if self.individual[i,j] < self.xmin:
                    if self.self.velocity[i,j] < 0:
                        self.velocity[i,j] = - self.velocity[i,j]
                    if self.individual[i,j] < 1.1*self.xmin:
                        self.individual[i,j] = np.random.uniform(self.xmin,self.xmax)
                
                if self.individual[i,j] > self.xmax:
                    if self.velocity[i,j] > 0:
                        self.velocity[i,j] = - self.velocity[i,j]
                    if self.individual[i,j] > 1.1*self.xmax:
                        self.individual[i,j] = np.random.uniform(self.xmin,self.xmax)
                    
            Data = self.Data
            self.likelihood[i] = Data.evolve(self.individual[i], self.prob_func)

            #If the likelihood is equal, increase self.velocity
            if any(abs(self.likelihood[i] - BM_val[i]) < [1e-3 for p in range(self.dim)]):
                self.velocity[i] = self.velocity[i] +A
            
            #If likelihood smaller or loudness over threshold -> Update BM and reduce loudness
            if self.likelihood[i] < BM_val[i] or np.random.uniform(0,1) < A:
                self.self.pulse[i] = self.pulse[i]*(1- np.exp(-self.gamma*gen))
                A = self.alfa*A
                BM[i] = self.individual[i]
                BM_val[i] = self.likelihood[i]           
        #Sort the bats
        sort = np.argsort(BM_val)
        BM = BM[sort]
        BM_val = BM_val[sort]        
        gen += 1
        
    
    def init_vel(self):
        role_pop = []
        kar = np.zeros((self.num_ind,self.dim))
        vel = np.zeros((self.num_ind,self.dim))
        puls = np.zeros((self.num_ind,self.dim))
        verdi =np.ones(self.num_ind)
        for i in range(self.num_ind):
            for j in range(self.dim):
                vel[i,j]    = np.random.uniform(-1,1)
                puls[i,j]   = self.rj
        self.velocity.append(vel)
        self.pulse.append(puls)
        sort = np.argsort(verdi)