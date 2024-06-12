import numpy as np
import sys
sys.path.append(r'C:\Users\Lenovo\Documents\Master')
from problem_func import * 

class jDE:
    def __init__(self, individual, likelihood, problem_func, xmin, xmax):
        self.problem_func   = problem_func
        self.individual     = individual
        self.likelihood     = likelihood
        self.xmin,self.xmax = xmin, xmax
        self.num_ind        = len(likelihood)
        self.dim            = len(individual[0])
        self.u              = np.zeros_like(self.individual)
        self.v              = np.zeros_like(self.individual)
        self.F_l, self.F_u  = 0.1, 0.9 
        self.Flist          = [np.random.uniform(0.1,1) for p in range(self.num_ind)]
        self.CRlist         = [np.random.uniform(0,0.99) for p in range(self.num_ind)]
        self.tau1,self.tau2 = 0.1,0.1
        self.Data           = Problem_Function(self.dim)
        self.nfe            = 0
        self.hist_data      = []

    def evolve(self):
        self.nfe        = 0
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
            rn = np.random.randint(0, self.dim)
            for j in range(self.dim):
                randint = np.random.uniform(0,1)
                if randint < self.CRlist[i] or rn == j:
                    self.u[i,j] = self.v[i,j]
                else:
                    self.u[i,j] = self.individual[i,j]
                
                self.u[i], status = self.check_oob(self.u[i])
                if status:
                    temp, true_likelihood = self.eval_likelihood_ind(self.u[i])
                    k = [self.u[i,j] for j in range(self.dim)] + [true_likelihood] + [1]
                    self.hist_data.append(k)    
                    self.nfe += 1
                    if temp < self.likelihood[i]:
                        self.individual[i] = self.u[i]
                        self.likelihood[i] = temp
                    
        #Adaptation
        for i in range(self.num_ind):
            ru1 = np.random.uniform(0,1)
            ru2 = np.random.uniform(0,1)
            ru3 = np.random.uniform(0,1)
            ru4 = np.random.uniform(0,1)
            if ru1 < self.tau1:
                self.Flist[i] += self.F_l + ru2 * self.F_u
            if ru3 < self.tau2:
                self.CRlist[i] = ru4
                
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

    #Metode for å evaluere likelihood til et enkelt individ.
    #Liker ikke helt å måtte kalle på den her også, men det er hittil det beste jeg har.
    def eval_likelihood_ind(self, individual):
        Data = self.Data
        return Data.evaluate(individual, self.problem_func)
