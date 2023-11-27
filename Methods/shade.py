import numpy as np
import sys
# sys.path.append(r'C:\Users\Lenovo\Documents\Master')
from problem_func import *



class SHADE:
    def __init__(self, individual, likelihood, problem_func):
        self.dim        = len(individual[0])
        self.individual = individual
        self.likelihood = likelihood
        self.num_ind    = len(likelihood)
        self.prob_func  = problem_func
        self.k_arg      = 0 
        
        p_i             = np.random.uniform(2/self.num_ind, 0.2)
        NP              = int(self.num_ind * p_i)        
        H               = self.num_ind
        
        self.A          = []
        self.Flist      = [0.1 for p in range(self.num_ind)]
        self.CRlist     = [0.1 for p in range(self.num_ind)]
        self.M_CR       = [0.1 for p in range(self.num_ind)]
        self.M_F        = [0.1 for p in range(self.num_ind)]
        
        self.Data = Problem_Function(self.dim)


    def evolve(self):
        S_CR       = []
        S_F        = []
        delta_f    = []

        self.u  = np.zeros_like(self.individual)
        self.v  = np.zeros_like(self.individual)
        sort    = np.argsort(self.likelihood, axis = 0)
        best_indexes    = sort[0:self.num_ind]
        xpbest          = self.individual[best_indexes]
        self.abs_best   = self.likelihood[best_indexes[0]]
        self.best_ind   = self.individual[best_indexes][0]
        #Mutant vector
        for i in range(self.num_ind-1):
            ri = np.random.randint(1,self.num_ind) 
            self.CRlist[i] = np.random.normal(self.M_CR[ri], 0.1)
            #Burde være Cauchy-fordeling
            self.Flist[i]  = np.random.normal(self.M_F[ri], 0.1)
            
            #Current to pbest/1 
            ri1 = np.random.randint(self.num_ind)
            ri2 = np.random.randint(self.num_ind)
            ri3 = np.random.randint(self.num_ind)
            
            self.v[i] = self.individual[i] + self.Flist[i]*(xpbest[ri3]-self.individual[i]) + self.Flist[i]*(self.individual[ri1]- self.individual[ri2])
                                            
        #Crossover
        for i in range(self.num_ind):
            for j in range(self.dim):
                randint = np.random.randint(0,1)
                if randint < self.CRlist [i]:
                    self.u[i,j] = self.v[i,j]
                
            temp = self.eval_likelihood_ind(self.u[i])     
            if temp <= self.likelihood[i]:
                self.individual[i] = self.u[i]
                self.A.append(self.individual[i])        
                delta_f.append(temp-self.likelihood[i])                
                S_CR.append(self.CRlist[i])
                S_F.append(self.Flist[i])
                self.individual[i] = self.u[i]
                self.likelihood[i] = temp
                if len(self.A) > self.num_ind :
                    del self.A[np.random.randint(0, self.num_ind)]
        #Update weights
        if len(S_CR) != 0:
            if self.k_arg>=self.num_ind:
                self.k_arg = 1
            wk = []
            mcr = 0
            mf_nom = 0
            mf_denom = 0
            tol = 1e-3
            for arg in range(len(S_CR)):
                wk.append(delta_f[arg]/(sum(delta_f)+tol))
            for arg in range(len(S_CR)):
                mcr += wk[arg] * S_CR[arg]
                mf_nom  += wk[arg]*S_F[arg]**2
                mf_denom += wk[arg]*S_F[arg]
            self.M_CR[self.k_arg] = mcr
            self.M_F[self.k_arg] = mf_nom/(mf_denom+tol)
            self.k_arg += 1

      
    #Metode for å evaluere likelihood til et enkelt individ.
    #Liker ikke helt å måtte kalle på den her også, men det er hittil det beste jeg har.
    def eval_likelihood_ind(self, individual):
        # Data = Problem_Function(self.dim)
        Data = self.Data
        return Data.evaluate(individual, self.prob_func)
        
