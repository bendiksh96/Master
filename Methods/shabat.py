import numpy as np
import sys
sys.path.append(r'C:\Users\Lenovo\Documents\Master')
from problem_func import *



class shabat:
    def __init__(self, individual, likelihood, problem_func, standard = True):
        self.dim        = len(individual[0])
        self.individual = individual
        self.likelihood = likelihood
        self.individual_bat = individual
        self.likelihood_bat = likelihood
        self.num_ind    = len(likelihood)
        self.prob_func  = problem_func
        self.k_arg      = 0

        self.Flist      = [0.1 for p in range(self.num_ind)]
        self.CRlist     = [0.1 for p in range(self.num_ind)]
        self.M_CR       = [0.1 for p in range(self.num_ind)]
        self.M_F        = [0.1 for p in range(self.num_ind)]

        self.Data = Problem_Function(self.dim)

        self.velocity   = np.zeros((self.num_ind, self.dim))
        self.pulse      = np.zeros((self.num_ind, self.dim))
        self.f_j        = np.zeros_like(likelihood)
        self.A          = np.zeros_like(likelihood)
        self.gen        = 0
        if standard == True:
            # self.rj          = .02
            self.rj0         = .2
            self.A_fac       = .03
            self.eps         = .01
            self.gamma       = .9
            self.alfa        = .7

        for i in range(self.num_ind):
            for j in range(self.dim):
                self.velocity[i,j]    = np.random.uniform(-1,1)
                self.pulse[i,j]       = np.random.uniform(0,1)
            self.f_j[i] = np.random.uniform(0, .5)
            self.A[i]   = self.A_fac

    def set_limits(self, xmin, xmax):
        self.xmin = xmin; self.xmax = xmax
    def evolve(self):

        #Reset success parameters
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
        # print(self.individual)
        #Crossover
        for i in range(self.num_ind-1):
            for j in range(self.dim):
                randint = np.random.randint(0,1)
                if randint < self.CRlist [i]:
                    # self.u[i,j] = self.v[i,j]
                    #har trikset endel her. Fikk null, men tror det er slik det må se ut
                    perceived_likelihood, true_likelihood  = self.eval_likelihood_ind(self.v[i])
                    if perceived_likelihood <= self.likelihood[i]:
                        self.individual[i] = self.v[i]
                        delta_f.append(perceived_likelihood-self.likelihood[i])
                        S_CR.append(self.CRlist[i])
                        S_F.append(self.Flist[i])
                        # print(self.v[i,j])
                        self.individual[i] = self.v[i]
                        self.likelihood[i] = true_likelihood

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



        for i in range(self.num_ind):
            if self.likelihood[i] < 3:
                self.individual[i], self.likelihood[i] = self.bat_me_up(self.individual[i], self.likelihood[i])

    def bat_me_up(self, best_ind, best_val):
        for k in range(self.num_ind):
            for j in range(self.dim):
                eps = np.random.uniform(-.01,.01)
                self.individual_bat[k,j] = best_ind[j] + eps*self.A[k]
            self.likelihood_bat[k], z = self.eval_likelihood_ind(self.individual_bat[k])

        iter = 0 ; max_iter = 5
        while iter < max_iter:
            sorty = np.argsort(self.f_j)
            low_freq   = self.f_j[sorty][0]
            high_freq  = self.f_j[sorty][-1]
            for k in range(self.num_ind):
                for j in range(self.dim):
                    #Frequency
                    self.f_j[k] = low_freq + (high_freq-low_freq) * np.random.uniform(0,1)
                    #Velocity
                    self.velocity[k,j] = self.velocity[k,j] +(self.individual[k,j] - best_ind[j]) * self.f_j[k]
                var = self.velocity[k] + self.individual[k]
                temp, z = self.eval_likelihood_ind(var)
                if temp < self.likelihood[k]:
                    self.individual_bat[k] = var

            sort = np.argsort(self.likelihood_bat)
            
            if self.likelihood_bat[sort][0]< best_val:
                best_ind = self.individual_bat[sort][0]
                best_val = self.likelihood_bat[sort][0]
                
            self.A *= self.A *0.8
            for k in range(self.num_ind):
                for j in range(self.dim):
                    eps = np.random.uniform(-1,1)
                    self.individual_bat[k,j] = best_ind[j] + eps*self.A[k]
                self.likelihood_bat[k], z = self.eval_likelihood_ind(self.individual_bat[k])
            iter += 1

        for i in range(self.num_ind):
            if self.likelihood_bat[i] < best_val:
                best_ind = self.individual_bat[i]
                best_val = self.likelihood_bat[i]
        return best_ind, best_val

    #Metode for å evaluere likelihood til et enkelt individ.
    #Liker ikke helt å måtte kalle på den her også, men det er hittil det beste jeg har.
    def eval_likelihood_ind(self, individual):
        # Data = Problem_Function(self.dim)
        Data = self.Data
        return Data.evaluate(individual, self.prob_func)

