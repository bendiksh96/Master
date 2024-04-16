import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r'C:\Users\Lenovo\Documents\Master')
from problem_func import *

class cbat:
    def __init__(self, individual, likelihood, problem_func, standard = True):
        self.individual = individual
        self.likelihood = likelihood
        self.true_likelihood = likelihood
        self.dim        = len(individual[0])
        self.num_ind    = len(likelihood)
        self.prob_func  = problem_func
        self.Data       = Problem_Function(self.dim)
        

        self.iter       = 0
        self.BM         = np.zeros_like(self.individual)
        self.BM_val     = np.zeros_like(self.likelihood)
        self.BM_val[:]  = 100
        self.velocity   = np.zeros((self.num_ind, self.dim))
        self.pulse      = np.zeros((self.num_ind, self.dim))
        self.f_j        = np.zeros_like(likelihood)
        self.A          = np.zeros_like(likelihood)
        self.gen        = 0
        
        self.hist_data = []
        if standard == True:
            # self.rj          = .02
            self.rj0         = .2
            self.A_fac       = .7
            self.eps         = .01
            self.gamma       = .9
            self.alfa        = .7

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
        self.nfe = 0
        sort_1      = np.argsort(self.likelihood)
        best_ind    = self.individual[sort_1][0]

        sort        = np.argsort(self.f_j)
        low_freq   = self.f_j[sort][0]
        high_freq  = self.f_j[sort][-1]

        for i in range(self.num_ind):
            for j in range(self.dim):
                #Frequency
                self.f_j[i] = low_freq + (high_freq-low_freq) * np.random.uniform(0,1)
                #Velocity
                self.velocity[i,j] = self.velocity[i,j] +(self.individual[i,j] - best_ind[j]) * self.f_j[i]
                if self.individual[i,j] < self.xmin:
                    if self.velocity[i,j] < 0:
                        self.velocity[i,j] = - self.velocity[i,j]
                    if self.individual[i,j] < 1.1*self.xmin:
                        self.individual[i,j] = np.random.uniform(self.xmin,self.xmax)

                if self.individual[i,j] > self.xmax:
                    if self.velocity[i,j] > 0:
                        self.velocity[i,j] = - self.velocity[i,j]
                    if self.individual[i,j] > 1.1*self.xmax:
                        self.individual[i,j] = np.random.uniform(self.xmin,self.xmax)
                
            var = self.velocity[i] + self.individual[i]
                   
            var, status = self.check_oob(var)
            if status:
                perceived_likelihood, true_likelihood  = self.eval_likelihood_ind(var) 
                k = [var[j] for j in range(self.dim)] + [true_likelihood] + [1]
                self.likelihood[i] = perceived_likelihood
                self.hist_data.append(k)
                self.nfe += 1  
            


            #If likelihood smaller or loudness over threshold -> Update BM and reduce loudness
            if self.likelihood[i] < self.BM_val[i] or np.random.uniform(0,1) < self.A[i]:
                self.pulse[i]   = self.pulse[i]*(1- np.exp(-self.gamma*self.gen))
                self.A[i]       = self.alfa*self.A[i]
                self.BM[i]      = self.individual[i]
                self.BM_val[i]  = self.likelihood[i]


        #øke hver likelihood om ingen bevegelse
        sort_2        = np.argsort(self.likelihood)
        best_ind      = self.individual[sort_2][0]
        
        for i in range(self.num_ind):
            r1 = np.random.uniform(0,1)
            if r1< .3:
                for j in range(self.dim):
                    eps = np.random.uniform(-1,1)
                    self.individual[i,j] = best_ind[j] + eps*self.A[i]
            else:
                r2 =  np.random.randint(0,self.num_ind)
                r3 =  np.random.randint(0,self.num_ind)
                r4 =  np.random.randint(0,self.num_ind)
                self.individual[i] = self.individual[r2] + .5* (self.individual[r3]-self.individual[r4])
            self.individual[i], status = self.check_oob(self.individual[i])
            if status:
                perceived_likelihood, true_likelihood  = self.eval_likelihood_ind(self.individual[i]) 
                k = [self.individual[i,j] for j in range(self.dim)] + [true_likelihood] + [1]
                self.likelihood[i] = perceived_likelihood                
                self.hist_data.append(k)
                self.nfe += 1  

                
        #Sort the bats
        self.true_likelihood    = self.likelihood
        sort                    = np.argsort(self.BM_val)
        self.BM                 = self.BM[sort]
        self.BM_val             = self.BM_val[sort]
        self.gen                += 1


    def init_vel(self):
        for i in range(self.num_ind):
            for j in range(self.dim):
                self.velocity[i,j]    = np.random.uniform(-1,1)
                self.pulse[i,j]       = np.random.uniform(0,1)
            self.f_j[i] = np.random.uniform(0, 2)
            self.A[i]   = self.A_fac

    def check_oob(self, candidate):
        candidate_status = True
        for j in range(self.dim):
            if candidate[j] < self.xmin:
                var = self.xmin - (candidate[j] - self.xmin)
                if var > self.xmax or var < self.xmin:
                    candidate_status = False
                else:
                    candidate[j] = var
                    
            if  candidate[j] > self.xmax:
                var  = self.xmax - (candidate[j] - self.xmax)
                if var < self.xmin or var > self.xmax:
                    candidate_status = False
                else:
                    candidate[j] = var
        return candidate, candidate_status

    def eval_likelihood_ind(self, individual):
        Data = self.Data
        # Data.param_change(0, 1.15)
        perceived_likelihood, true_likelihood = Data.evaluate(individual, self.prob_func)
        return perceived_likelihood, true_likelihood
