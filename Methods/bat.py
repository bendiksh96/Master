import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r'C:\Users\Lenovo\Documents\Master')
from problem_func import *

class Bat:
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

        self.gen        = 0
        if standard == True:
            self.rj          = .02
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
        sort        = np.argsort(self.likelihood)
        best_ind    = self.individual[sort][0]
        best_val    = self.likelihood[sort][0]
        worst_val   = self.likelihood[sort][-1]

        for i in range(self.num_ind):
            for j in range(self.dim):
                u = np.random.uniform(0,1)
                #Frequency
                # f_j = best_val + (best_val-worst_val) * np.random.uniform(0,1)
                
                f_j = best_val + (best_val-worst_val) * np.random.uniform(0,1)/worst_val
                
                # f_j = (worst_val)/(worst_val-best_val) * np.random.uniform(-1,1)
                
                # f_j = (best_val + np.mean(self.likelihood))/worst_val * np.random.uniform(-1,1)

                # f_j = (abs(worst_val-np.mean(self.likelihood)))/worst_val * np.random.uniform(-1,1)

                # print('f_j:', f_j)
                
                self.velocity[i,j] = self.velocity[i,j] +(self.individual[i,j] - best_ind[j]) * f_j
                # print('vel:', self.velocity[i,j])
                # print()

            # exit()
                
                if u < self.pulse[i,j]:
                    self.individual[i,j] = best_ind[j] + self.eps*self.A
                    #Her er eps .01, A <= .5
                    #alle pulsene er 0.2
                    #dvs. at i 20% av tilfellene blir ind best_ind + 0.05
                    #Hva med:
                        # self.individual[i,j] = self.individual + self.A*np.random.uniform(-1,1)

                else:
                    self.individual[i,j] = self.velocity[i,j] + self.individual[i,j]

                #oob
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
            self.likelihood[i], self.true_likelihood[i] = self.eval_likelihood_ind(self.individual[i])

            #If the likelihood is equal, increase self.velocity
            #Tror ikke dette trengs, da alle flaggiser oppdateres
            # if (abs(self.likelihood[i] - self.BM_val[i]) < 1e-3):
            #     self.velocity[i] = self.velocity[i] +self.A

            #If likelihood smaller or loudness over threshold -> Update BM and reduce loudness
            if self.likelihood[i] < self.BM_val[i] or np.random.uniform(0,1) < self.A:
                # print('puls_faktor:',(1- np.exp(-self.gamma*self.gen)))
                self.pulse[i]   = self.pulse[i]*(1- np.exp(-self.gamma*self.gen))
                self.A          = self.alfa*self.A
                self.BM[i]      = self.individual[i]
                self.BM_val[i]  = self.likelihood[i]
                
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
                self.pulse[i,j]       = self.rj


    def eval_likelihood_ind(self, individual):
        Data = self.Data
        perceived_likelihood, true_likelihood = Data.evaluate(individual, self.prob_func)
        return perceived_likelihood, true_likelihood
