import sys
from copy import deepcopy
import numpy as np
import random as rd
from vis_writ import *

# sys.path.append(r'C:\Users\Lenovo\Documents\Master\Methods')
from problem_func import *
from Methods.jde import *
from Methods.bat import *
from Methods.shade import *
from Methods.d_shade import *


class DEVO_class:
    def __init__(self, dim, problem_func, method):
        self.problem_func   = problem_func
        self.method         = method
        self.dim            = dim
        self.hist_ind       = []
        self.hist_lik       = []
        self.iter           = 0
        self.nfe            = 0

    #Initialize population(s)        
    def intialize_population(self, xmin, xmax, num_ind):
        self.xmin,self.xmax = xmin,xmax
        self.num_ind        = num_ind
        self.individual     = np.ones((self.num_ind, self.dim))
        self.likelihood     = np.ones((self.num_ind))

        #Initialize every individuals position and likelihood
        Data = Problem_Function(self.dim)
        for i in range(self.num_ind):
            for j in range(self.dim):
                self.individual[i,j] = np.random.uniform(xmin, xmax)
            self.likelihood[i] = Data.evaluate(self.individual[i,:], self.problem_func)
        self.nfe += self.num_ind
    
    def evolve(self, maxiter, tol=1e-3):
        if self.method == 'jde':
            iter_likelihood = []; conv = False
            mod = jDE(self.individual, self.likelihood, self.problem_func)
            while self.iter < maxiter and conv == False:
                mod.evolve()
                self.check_oob()
                self.nfe  += self.num_ind
                self.iter +=1

                if self.iter > 10:
                    current_std = np.std(mod.likelihood)
                    atol = tol
                    threshold_std = atol + tol * np.abs(np.mean(mod.likelihood))
                    print(f"DEBUG: current_std: {current_std}   threshold_std: {threshold_std}")
                    if current_std <= threshold_std:
                        print('Converged!')
                        conv = True

                # if self.iter > 10:
                #     # _Anders
                #     # if (np.mean(iter_likelihood)-iter_likelihood[-1]) < tol:
                #     current_eps = abs(np.mean(iter_likelihood)-np.min(iter_likelihood))
                #     print(f"DEBUG: current_eps = {current_eps}")
                #     if current_eps < tol:
                #         print('Converged!')
                #         conv = True
                #     del iter_likelihood[0]

                for i in range(self.num_ind):
                    # _Anders
                    self.hist_ind.append(deepcopy(self.individual[i]))
                    self.hist_lik.append(deepcopy(self.likelihood[i]))


        if self.method == 'shade':
            iter_likelihood = []; conv = False
            mod = SHADE(self.individual, self.likelihood, self.problem_func)
            while self.iter < maxiter and conv == False:
                mod.evolve()
                self.check_oob()
                
                self.nfe  += self.num_ind
                self.iter +=1
                iter_likelihood.append(np.mean(mod.likelihood))
                if self.iter > 10:
                    # _Anders
                    # if (np.mean(iter_likelihood)-iter_likelihood[-1]) < tol:
                    current_eps = abs(np.mean(iter_likelihood)-np.min(iter_likelihood))
                    print(f"DEBUG: current_eps = {current_eps}")
                    if current_eps < tol:
                        print('Converged!')
                        conv = True
                    del iter_likelihood[0]
                for i in range(self.num_ind):
                    self.hist_ind.append(deepcopy(self.individual[i]))
                    self.hist_lik.append(deepcopy(self.likelihood[i]))

                    
        
        if self.method == 'ddms':
            pass
        
        if self.method == 'bat':
            iter_likelihood = []; conv = False
            mod = Bat(self.individual, self.likelihood, self.problem_func)
            mod.set_limits(self.xmin, self.xmax)
            while self.iter < maxiter and conv == False:
                mod.evolve()
                self.check_oob()
                
                self.nfe  += self.num_ind
                self.iter +=1
                iter_likelihood.append(np.mean(mod.likelihood))
                if self.iter > 10:
                    if (np.mean(iter_likelihood)-iter_likelihood[-1]) < tol:
                        print('Converged!')
                        conv = True
                    del iter_likelihood[0]
                for i in range(self.num_ind):
                    self.hist_ind.append(deepcopy(self.individual[i]))
                    self.hist_lik.append(deepcopy(self.likelihood[i]))

        
        if self.method == 'double_shade':
            iter_likelihood = []; conv = False
            mod = d_SHADE(self.individual, self.likelihood, self.problem_func)
            while self.iter < maxiter and conv == False:
                mod.evolve_converge()
                self.check_oob()
                
                self.nfe  += self.num_ind
                self.iter += 1
                iter_likelihood.append(np.mean(mod.likelihood))
                
                if self.iter > 10:
                    # print(np.mean(iter_likelihood)-iter_likelihood[-1])
                    if (np.mean(iter_likelihood)-iter_likelihood[-1]) < tol:
                        print('Converged!')
                        
                        conv = True
                    del iter_likelihood[0]
                for i in range(self.num_ind):
                    self.hist_ind.append(deepcopy(self.individual[i]))
                    self.hist_lik.append(deepcopy(self.likelihood[i]))
            if conv:
                #Change function
                mod.prob_func = 'mod_' + self.problem_func
                best = mod.abs_best
                mod.Data.param_change(best=best, delta=.1,sigma=1)
                self.intialize_population(self.xmin, self.xmax, self.num_ind)
                while self.iter < maxiter:
                    mod.evolve_explore()
                    
                    self.check_oob()
                    
                    self.nfe  += self.num_ind
                    self.iter += 1
                    #Run indefinetly?                    
                    for i in range(self.num_ind):
                        self.hist_ind.append(deepcopy(self.individual[i]))
                        self.hist_lik.append(deepcopy(self.likelihood[i]))

        
        
    #Sjekker rammebetingelser
    def check_oob(self):
        var = 0
        for i in range(self.num_ind):
            for j in range(self.dim):
                if  self.individual[i,j] < self.xmin:
                    var = self.xmin - (self.individual[i,j] - self.xmin)
                    if var > self.xmax:
                        self.individual[i,j] = 0
                        print('Some error has occured in oob')
                    else:
                        self.individual[i,j] = var

                if  self.individual[i,j] > self.xmax:
                    var  = self.xmax - (self.individual[i,j] - self.xmax)
                    if var < self.xmin:
                        self.individual[i,j] = 0
                        print('Some error has occured in oob')
                    else:
                        self.individual[i,j] = var

        
