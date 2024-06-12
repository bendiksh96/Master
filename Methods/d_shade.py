import numpy as np
import sys
sys.path.append(r'C:\Users\Lenovo\Documents\GitHub\Master')
from problem_func import *

class d_SHADE:   
    def __init__(self, individual, likelihood, problem_func, xmin, xmax):
        self.dim        = len(individual[0])
        self.individual = individual
        self.likelihood = likelihood
        self.xmin, self.xmax = xmin, xmax
        self.true_likelihood = likelihood
        self.hist_data  = []
        self.nfe        = 0
        
        self.num_ind    = len(likelihood)
        self.prob_func  = problem_func
        self.k_arg      = 0 
        
        self.optimum    = 10
        
        self.optimal_individual    = np.zeros_like(self.individual)
        
        self.A          = []
        self.Flist      = [0.1 for p in range(self.num_ind)]
        self.CRlist     = [0.1 for p in range(self.num_ind)]
        self.M_CR       = [0.5 for p in range(self.num_ind)]
        self.M_F        = [0.5 for p in range(self.num_ind)]
        
        self.Data = Problem_Function(self.dim)


    def evolve_converge(self):
        self.nfe   = 0 
        S_CR       = []
        S_F        = []
        delta_f    = []

        self.v          = np.zeros_like(self.individual)
        sort            = np.argsort(self.likelihood, axis = 0)
        best_indexes    = sort[0:self.num_ind]
        xpbest          = self.individual[best_indexes]
        self.abs_best   = self.likelihood[best_indexes[0]]
        self.best_ind   = self.individual[best_indexes][0]
        #Mutant vector
        for i in range(self.num_ind):
            ri = np.random.randint(1,self.num_ind) 

            self.CRlist[i] = np.random.normal(self.M_CR[ri], 0.1)
            cauchy = np.random.standard_cauchy()
            self.Flist[i]  = self.M_F[ri] + 0.1*cauchy 
            
            #Current to pbest/2 
            ri1 = np.random.randint(self.num_ind)
            ri2 = np.random.randint(self.num_ind)
            ri3 = np.random.randint(self.num_ind/4)
            
            self.v[i] = self.individual[i] + self.Flist[i]*(xpbest[ri3]-self.individual[i]) + self.Flist[i]*(self.individual[ri1]- self.individual[ri2])
                                            
        #Crossover
        for i in range(self.num_ind):
            randint = np.random.uniform(0,1)
            if randint < self.CRlist[i]:
                self.v[i], status = self.check_oob(self.v[i])
                if status:
                    perceived_likelihood, true_likelihood  = self.eval_likelihood_ind(self.v[i])    
                    k = [self.v[i,j] for j in range(self.dim)] + [true_likelihood] + [1]
                    self.hist_data.append(k)
                    self.nfe += 1
 
                    if perceived_likelihood <= self.likelihood[i]:
                        self.individual[i] = self.v[i]
                        self.A.append(self.individual[i])        
                        delta_f.append(perceived_likelihood-self.likelihood[i])                
                        S_CR.append(self.CRlist[i])
                        S_F.append(self.Flist[i])
                        self.likelihood[i] = perceived_likelihood
                        
                        if len(self.A) > self.num_ind :
                            del self.A[np.random.randint(0, self.num_ind)]
        #Update weights
        if len(S_CR) != 0:
            wk = []; mcr = 0;mf_nom = 0;mf_denom = 0;tol = 1e-3
            if self.k_arg>=self.num_ind:
                self.k_arg = 1
            
            for arg in range(len(S_CR)):
                wk.append(delta_f[arg]/(sum(delta_f)+tol))
            for arg in range(len(S_CR)):
                mcr += wk[arg] * S_CR[arg]
                mf_nom  += wk[arg]*S_F[arg]**2
                mf_denom += wk[arg]*S_F[arg]
            self.M_CR[self.k_arg] = mcr
            self.M_F[self.k_arg] = mf_nom/(mf_denom+tol)
            self.k_arg += 1

    def evolve_explore(self):
        self.nfe = 0
        #Initialize the list of weights
        S_CR       = []
        S_F        = []
        delta_f    = []

        #Initialize temporary variables
        self.v          = np.zeros_like(self.individual)
        
        #Previous best indices
        sort            = np.argsort(self.likelihood, axis = 0)
        best_indexes    = sort[0:self.num_ind]
        xpbest          = self.individual[best_indexes]
        self.abs_best   = self.likelihood[best_indexes[0]]
        self.best_ind   = self.individual[best_indexes][0]

        #Evolve individuals
        for i in range(self.num_ind):            
            ri = np.random.randint(1,self.num_ind) 
            self.CRlist[i] = np.random.normal(self.M_CR[ri], 0.1)
            cauchy = np.random.standard_cauchy()
            self.Flist[i]  = self.M_F[ri] + 0.1*cauchy 
            
            #pbest-to-pbest/1:
            ri1 = np.random.randint(self.num_ind)
            rii = np.random.randint(self.num_ind/2)
            ri3 = np.random.randint(self.num_ind/2)
            self.v[i] = xpbest[rii] + self.Flist[i]*(xpbest[ri3]-self.individual[ri1])
                                
        #Crossover
        
        for i in range(self.num_ind):
            randu = np.random.uniform(0,1)
            if randu < self.CRlist[i]:
                self.v[i], status = self.check_oob(self.v[i])
                #Check if the new crossover individual is superior to the prior
                if status:
                    perceived_likelihood, true_likelihood  = self.eval_likelihood_ind(self.v[i])     
                    k = [self.v[i,j] for j in range(self.dim)] + [true_likelihood] + [2]
                    self.hist_data.append(k)
                    self.nfe += 1
                    
                    if perceived_likelihood <= self.likelihood[i]:
                        self.individual[i] = self.v[i]
                        self.A.append(self.individual[i])        
                        delta_f.append(perceived_likelihood-self.likelihood[i])                
                        S_CR.append(self.CRlist[i])
                        S_F.append(self.Flist[i])
                        self.likelihood[i] = perceived_likelihood
                        self.true_likelihood[i] = true_likelihood

                        #If archive exceeds number of individuals, delete a random archived log.
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
        return Data.evaluate(individual, self.prob_func)
        
