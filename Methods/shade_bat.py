import numpy as np
import sys
sys.path.append(r'C:\Users\Lenovo\Documents\Master')
from problem_func import *



class shade_bat:
    def __init__(self, individual, likelihood, problem_func):
        self.dim        = len(individual[0])
        self.individual = individual
        self.likelihood = likelihood
        self.true_likelihood = likelihood
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

    def change_pop_size(self, num_ind):
        self.true_likelihood = np.zeros(num_ind)
    def evolve_converge(self):
        S_CR       = []
        S_F        = []
        delta_f    = []

        self.u          = np.zeros_like(self.individual)
        self.v          = np.zeros_like(self.individual)
        sort            = np.argsort(self.likelihood, axis = 0)
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
                
                    perceived_likelihood, true_likelihood  = self.eval_likelihood_ind(self.u[i])     
                    if perceived_likelihood <= self.likelihood[i]:
                        self.individual[i] = self.u[i]
                        self.A.append(self.individual[i])        
                        delta_f.append(perceived_likelihood-self.likelihood[i])                
                        S_CR.append(self.CRlist[i])
                        S_F.append(self.Flist[i])
                        self.individual[i] = self.u[i]
                        self.likelihood[i] = perceived_likelihood
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


    def initialize_bat(self, standard = True):
        self.iter       = 0        
        self.BM         = np.zeros((self.num_ind, self.dim))
        self.BM_val     = np.zeros(self.num_ind)
        self.BM_val[:]  = 100
        self.f_j        = np.zeros_like(self.likelihood)
        self.A          = np.zeros_like(self.likelihood)

        self.velocity   = np.zeros((self.num_ind, self.dim))
        self.pulse      = np.zeros((self.num_ind, self.dim))
        self.gen        = 0
        if standard == True:
            # self.rj          = .2
            self.rj0         = .2
            self.A_fac       = 1.4
            self.eps         = .01
            self.gamma       = .9
            self.alfa        = .9
        self.init_vel()

    def evolve_bat(self):           

        sort        = np.argsort(self.likelihood)
        best_ind    = self.individual[sort][0]

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
                   
            temp, z = self.eval_likelihood_ind(var)
            if temp < self.likelihood[i]:
                self.individual[i] = var 
            


            #If likelihood smaller or loudness over threshold -> Update BM and reduce loudness
            if self.likelihood[i] < self.BM_val[i] or np.random.uniform(0,1) < self.A[i]:
                # print('puls_faktor:',(1- np.exp(-self.gamma*self.gen)))
                self.pulse[i]   = self.pulse[i]*(1- np.exp(-self.gamma*self.gen))
                self.A[i]       = self.alfa*self.A[i]
                self.BM[i]      = self.individual[i]
                self.BM_val[i]  = self.likelihood[i]

        sort        = np.argsort(self.likelihood)
        best_ind    = self.individual[sort][0]
        for i in range(self.num_ind):
            for j in range(self.dim):
                eps = np.random.uniform(-1,1)
                self.individual[i,j] = best_ind[j] + eps * self.A[i]
            self.likelihood[i], z = self.eval_likelihood_ind(self.individual[i]) 
                
        #Sort the bats
        self.true_likelihood    = self.likelihood
        sort                    = np.argsort(self.BM_val)
        self.BM                 = self.BM[sort]
        self.BM_val             = self.BM_val[sort]
        self.gen                += 1


        # sort = np.argsort(self.likelihood)
        # best_ind = self.individual[sort][0]
        # best_val = self.likelihood[sort][0]
        # worst_val = self.likelihood[sort][-1]
        
        # for i in range(self.num_ind):
        #     for j in range(self.dim):
        #         u = np.random.uniform(0,1)
        #         #Frequency
        #         f_j = best_val + (best_val-worst_val) * np.random.uniform(0,1)
        #         self.velocity[i,j] = self.velocity[i,j] +(self.individual[i,j] - best_ind[j]) * f_j
        #         #Update self.individual if
        #         if u < self.pulse[i,j]:
        #             self.individual[i,j] = best_ind[j] + self.eps*self.A
        #         else:
        #             self.individual[i,j] = self.velocity[i,j] + self.individual[i,j] 
                
                
        #         #oob
        #         if self.individual[i,j] < self.xmin:
        #             if self.velocity[i,j] < 0:
        #                 self.velocity[i,j] = - self.velocity[i,j]
        #             if self.individual[i,j] < 1.2*self.xmin:
        #                 self.individual[i,j] = np.random.uniform(self.xmin,self.xmax)
                
        #         if self.individual[i,j] > self.xmax:
        #             if self.velocity[i,j] > 0:
        #                 self.velocity[i,j] = - self.velocity[i,j]
        #             if self.individual[i,j] > 1.2*self.xmax:
        #                 self.individual[i,j] = np.random.uniform(self.xmin,self.xmax)
                    
        #     perceived_likelihood, self.true_likelihood[i] = self.eval_likelihood_ind(self.individual[i])        
        #     self.likelihood[i] = perceived_likelihood
            
        #     #If the likelihood is equal for any individual, increase its velocity
        #     if abs(self.likelihood[i] - self.BM_val[i]) < 1e-3:
        #         self.velocity[i] = self.velocity[i] + self.A*1e-2
            
        #     #If likelihood smaller or loudness over threshold -> Update BM and reduce loudness
        #     if self.likelihood[i] < self.BM_val[i] or np.random.uniform(0,1) < self.A:
        #         self.pulse[i] = self.pulse[i]*(1- np.exp(-self.gamma*self.gen))
        #         self.A = self.alfa*self.A
        #         self.BM[i] = self.individual[i]
        #         self.BM_val[i] = self.likelihood[i]           
        # #Sort the bats
        # sort = np.argsort(self.BM_val)
        # self.BM = self.BM[sort]
        # self.BM_val = self.BM_val[sort]        
        # self.gen += 1        
    
    def set_limits(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
    
    def init_vel(self):
        for i in range(self.num_ind):
            for j in range(self.dim):
                self.velocity[i,j]    = np.random.uniform(-1,1)/5
                self.pulse[i,j]       = self.rj
            self.f_j[i] = np.random.uniform(0, 2)
            self.A[i]   = self.A_fac

    def eval_likelihood_ind(self, individual):
        Data = self.Data
        perceived_likelihood, true_likelihood = Data.evaluate(individual, self.prob_func)
        return perceived_likelihood, true_likelihood

    def change_var(self, rj, rj0, A, eps, gamma, alfa):
        self.rj          = rj
        self.rj0         = rj0
        self.A           = A
        self.eps         = eps
        self.gamma       = gamma
        self.alfa        = alfa
