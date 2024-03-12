import numpy as np
import sys
sys.path.append(r'C:\Users\Lenovo\Documents\Master')
from problem_func import * 

    
class jDErpo:
    def __init__(self, individual, likelihood, problem_func):
        self.problem_func   = problem_func
        self.individual     = individual
        self.likelihood     = likelihood
        self.true_likelihood= likelihood
        self.num_ind        = len(likelihood)
        self.dim            = len(individual[0])
        self.u              = np.zeros_like(self.individual)
        self.v              = np.zeros_like(self.individual)
        p_i                 = np.random.uniform(2/self.num_ind, 0.2)
        self.Flist          = [0.1 for p in range(self.num_ind)]
        self.Fupp           = 1
        self.CRlist         = [0.1 for p in range(self.num_ind)]
        self.CRupp          = 1
        self.tau1,self.tau2 = 0.1,0.1
        self.nfe            = 0 
        self.nfe_tot        = 0 
        self.max_nfe        = 0
        self.hist_data      = []
        self.Data           = Problem_Function(self.dim)
    
    def evolve(self):
        self.nfe = 0 
        sort_index = np.argsort(self.likelihood, axis = 0)
        self.abs_best = self.likelihood[sort_index][0]
        self.at         =self.individual[sort_index][0]
        
        NP = int(self.num_ind * np.random.uniform(0,1)+1)

        self.u = np.zeros_like(self.individual)
        self.v = np.zeros_like(self.individual)
        best_indexes = sort_index[0:NP]
        xpbest = self.individual[best_indexes]
        
        Flow  = 0.2  + 0.3  * self.nfe_tot/self.max_nfe
        CRlow = 0.05 + 0.95 * self.nfe_tot/self.max_nfe
        
        #Mutant vector
        for i in range(self.num_ind):
            ri1 = np.random.randint(self.num_ind)
            ri2 = np.random.randint(self.num_ind)
            ri3 = np.random.randint(self.num_ind)
            ri4 = np.random.randint(NP)
            if np.random.uniform(0,1) < 0.8 and self.nfe_tot > 0.8*self.max_nfe:
                #rand/1 scheme
                self.v[i] = self.individual[ri1] + self.Flist[i]* (self.individual[ri2]- self.individual[ri3])
            else:   
                #pBest/1 scheme
                self.v[i] = xpbest[ri4] + self.Flist[i]*(self.individual[ri2]- self.individual[ri3])
        
                
        for i in range(self.num_ind):
            randint = np.random.uniform(0,1)
            
            if randint < self.CRlist[i]:
                temp_likelihood, true_likelihood = self.eval_likelihood_ind(self.v[i])
                k = [self.v[i,j] for j in range(self.dim)] + [true_likelihood] + [1]
                self.hist_data.append(k)
                self.nfe += 1
                if  temp_likelihood < self.likelihood[i]:
                    
                    self.individual[i] = self.v[i]
                    self.likelihood[i] = temp_likelihood
                    self.true_likelihood[i] = true_likelihood
        for i in range(self.num_ind):
            ru1 = np.random.uniform(0,1)
            ru2 = np.random.uniform(0,1)
            ru3 = np.random.uniform(0,1)
            ru4 = np.random.uniform(0,1)
            if ru1 < self.tau1:
                self.Flist[i] =Flow + ru2*(self.Fupp - Flow)
            if ru3 < self.tau2:
                self.CRlist[i] = CRlow + ru4*(self.CRupp - CRlow)

    def eval_likelihood_ind(self, individual):
        # Data = Problem_Function(self.dim)
        Data = self.Data
        return Data.evaluate(individual, self.problem_func)
