import sys
import numpy as np
import random as rd
import csv
# sys.path.append(r'C:\Users\Bendik Selvaag-Hagen\OneDrive - Universitetet i Oslo\Documents\GitHub\Master')
sys.path.append(r'C:\Users\Lenovo\Documents\Master\Methods')

from vis_writ import *
# sys.path.append(r'C:\Users\Bendik Selvaag-Hagen\OneDrive - Universitetet i Oslo\Documents\GitHub\Master\Methods')
sys.path.append(r'C:\Users\Bendik Selvaag-Hagen\Desktop\Skole\Master\Methods')
from problem_func import *
from jde import *
from bat import *
from ddms import * 
from shade import *
from shabat import *
from jderpo import *
from d_shade import *
from shade_bat import *
from chaotic_bat import *
from d_shade_pso import * 
from d_shade_bat import * 

def loading_bar():
    toolbar_width = 40  # adjust the width of the loading bar
    total_time = 5  # specify the total time for the process completion
    
    elapsed_time = time.time() - start_time
    completed = int(elapsed_time / total_time * toolbar_width)
    remaining = toolbar_width - completed
    
    loading_bar = '[' + '=' * completed + '.' * remaining + ']'
    
    print(f'\rLoading: {loading_bar} {int(elapsed_time/total_time*100)}%', end='')
    
def conditions(problem_function):
    if problem_function == 'Himmelblau':
        xmin, xmax = -5,5
    if problem_function == 'Eggholder':
        xmin, xmax = -512,512
    if problem_function == 'Rosenbrock':
        xmin, xmax = -5,5
    return xmin, xmax

class DEVO_class:
    def __init__(self, dim, problem_func, method, log_thresh):
        self.problem_func   = problem_func
        self.method         = method
        self.dim            = dim
        self.hist_data      = []
        self.nfe            = 0
        self.best           = 10
        self.log_thresh     = log_thresh
        self.open_data()
        
    #Initialize population(s)        
    def initialize_population(self, xmin, xmax, num_ind):
        self.xmin,self.xmax     = xmin,xmax
        self.num_ind            = num_ind
        self.individual         = np.ones((self.num_ind, self.dim))
        self.likelihood         = np.ones((self.num_ind))
        self.actual_likelihood  = np.ones_like(self.likelihood)
        self.conv_iter          = 0
        
        #Initialize every individuals position and likelihood
        Data    = Problem_Function(self.dim)
        # Data.param_change(0, 1.15)
        for i in range(self.num_ind):
            var     = []
            for j in range(self.dim):
                self.individual[i,j] = np.random.uniform(xmin, xmax)
                var.append(self.individual[i,j])
            temp, z = Data.evaluate(self.individual[i,:], self.problem_func)
            self.likelihood[i] = temp
            var.append(temp)
            var.append(1)
            self.hist_data.append(var)
        self.nfe += self.num_ind
        
        self.write_data(self.hist_data)
        self.hist_data = []
    
    def evolve(self, max_nfe):
        self.max_nfe   = max_nfe
        if self.method == 'jde':
            b = 1
            iter_likelihood = []; tol = 1e-3; conv = False
            mod = jDE(self.individual, self.likelihood, self.problem_func)            
            while self.nfe < self.max_nfe and conv == False:
                mod.evolve()
                self.check_oob()
                self.nfe  += mod.nfe
                iter_likelihood.append(np.mean(mod.likelihood))
                if int(b*1e3)<self.nfe:
                    print('nfe:',self.nfe)
                    b+=1
                # for i in range(self.num_ind):
                #     var = []
                #     for j in range(self.dim):
                #         var.append(self.individual[i,j])
                #     var.append(self.likelihood[i])
                #     self.hist_data.append(var)
                # self.write_data()

                if len(self.hist_data)>int(1e5):
                    self.write_data(mod.hist_data)
                    mod.hist_data       = []

                if len(self.hist_data)>int(1e5):
                    self.write_data(mod.hist_data)
                    mod.hist_data       = []
            self.write_data(mod.hist_data)

        if self.method == 'jderpo':
            iter_likelihood = []; tol = 1e-3; conv = False
            mod = jDErpo(self.individual, self.likelihood, self.problem_func)
            mod.max_nfe = self.max_nfe
            b = 1
            while self.nfe < self.max_nfe and conv == False:
                mod.evolve()
                self.check_oob()
                self.nfe  += mod.nfe
                iter_likelihood.append(np.mean(mod.likelihood))
                if self.nfe > 10*self.num_ind:
                    if any(self.likelihood) == 0:
                        print('Conv')    
                        conv = True
                    del iter_likelihood[0]
                if int(b*1e3)<self.nfe:
                    print('nfe:',self.nfe)
                    b+=1

                # for i in range(self.num_ind):
                #     var = []
                #     for j in range(self.dim):
                #         var.append(self.individual[i,j])
                #     var.append(self.likelihood[i])
                #     self.hist_data.append(var)
                if len(mod.hist_data)>int(1e5):
                    self.write_data(mod.hist_data)
                    mod.hist_data       = []
            self.write_data(mod.hist_data)


        if self.method == 'shade':
            b = 1
            iter_likelihood = []; tol = 1e-3; conv = False
            mod = SHADE(self.individual, self.likelihood, self.problem_func)
            while self.nfe < self.max_nfe and conv == False:
                #Evolve the algorithm
                mod.evolve()
                self.check_oob()
                self.nfe  += mod.nfe

                if int(b*1e3)<self.nfe:
                    print('nfe:',self.nfe)
                    b+=1

                #Add individuals to history
                # for i in range(self.num_ind):
                #     var = []
                #     for j in range(self.dim):
                #         var.append(np.round(self.individual[i,j], 6))
                #     var.append(np.round(self.likelihood[i], 6))
                    # self.hist_data.append(var)
                #If history too large, write to file and reset
                if len(mod.hist_data)>int(1e5):
                    self.write_data(mod.hist_data)
                    mod.hist_data = []
            self.write_data(mod.hist_data)
                     
        if self.method == 'shabat':
            iter_likelihood = []; tol = 1e-3; conv = False
            mod = shabat(self.individual, self.likelihood, self.problem_func)
            mod.set_limits(self.xmin, self.xmax)
            while self.nfe < self.max_nfe and conv == False:
                mod.evolve()
                self.check_oob()
                
                self.nfe  += self.num_ind
                iter_likelihood.append(np.mean(mod.likelihood))
            
                for i in range(self.num_ind):
                    var = []
                    for j in range(self.dim):
                        var.append(self.individual[i,j])
                    var.append(self.likelihood[i])
                    self.hist_data.append(var)
                if len(self.hist_data)>int(1e5):
                    self.write_data()
                    self.hist_data       = []
            self.write_data()
            
        if self.method == 'ddms':
            iter_likelihood = []; tol = 1e-3; conv = False
            mod = DDMS(self.individual, self.likelihood, self.problem_func)
            mod.nfe_max = maxiter*self.num_ind
            while self.iter < maxiter and conv == False:
                mod.evolve()
                self.check_oob()
                
                self.nfe  += self.num_ind
                mod.nfe   += self.nfe
                self.iter += 1
                mod.gen   += 1

                iter_likelihood.append(np.mean(mod.likelihood))
                for i in range(self.num_ind):                    
                    self.hist_ind[self.iter] = self.individual[i]
                    self.hist_lik[self.iter] = self.likelihood[i]
        
        if self.method == 'bat':
            iter_likelihood = []; tol = 1e-5; conv = False
            mod = Bat(self.individual, self.likelihood, self.problem_func)
            mod.set_limits(self.xmin, self.xmax)
            self.iter_likelihood_mean = []
            self.iter_likelihood_best = []
            self.iter_likelihood_median = []
            
            while self.nfe < self.max_nfe and conv == False:
                mod.evolve()
                self.check_oob()
                
                self.nfe  += self.num_ind
                iter_likelihood.append(np.mean(mod.likelihood))
                
                for i in range(self.num_ind):
                    var = []
                    for j in range(self.dim):
                        var.append(self.individual[i,j])
                    var.append(self.likelihood[i])
                    self.hist_data.append(var)
                if len(self.hist_data)>int(1e5):
                    self.write_data()
                    self.hist_data       = []
            self.write_data()
        
        if self.method == 'chaotic_bat':
            iter_likelihood = []; tol = 1e-5; conv = False
            mod = cbat(self.individual, self.likelihood, self.problem_func)
            mod.set_limits(self.xmin, self.xmax)
            self.iter_likelihood_mean = []
            self.iter_likelihood_best = []
            self.iter_likelihood_median = []
            
            while self.nfe < self.max_nfe and conv == False:
                mod.evolve()
                self.check_oob()
                
                self.nfe  += 2*self.num_ind
                iter_likelihood.append(np.mean(mod.likelihood))
                
                for i in range(self.num_ind):
                    var = []
                    for j in range(self.dim):
                        var.append(self.individual[i,j])
                    var.append(self.likelihood[i])
                    #     var.append(mod.BM[i,j])
                    # var.append(mod.BM_val[i])
                    self.hist_data.append(var)
                if len(self.hist_data)>int(1e5):
                    self.write_data()
                    self.hist_data       = []
            self.write_data()

        if self.method == 'double_shade':
            iter_likelihood = []; tol = 1e-3; conv = False
            mod = d_SHADE(self.individual, self.likelihood, self.problem_func)
            while self.nfe < self.max_nfe and conv == False:
                mod.evolve_converge()
                self.check_oob()
                
                self.nfe  += self.num_ind
                iter_likelihood.append(np.mean(mod.likelihood))
                
                if int(self.nfe/self.num_ind) > 10:
                    # print(np.mean(iter_likelihood)-iter_likelihood[-1])
                    if (np.mean(iter_likelihood)-iter_likelihood[-1]) < tol or any(self.likelihood) == 0:
                        print('Conv')    
                        for i in range(self.num_ind):
                            var = []
                            for j in range(self.dim):
                                var.append(self.individual[i,j])
                            var.append(self.likelihood[i])
                            self.hist_data.append(var)

                        conv = True
                        self.conv_iter = self.nfe
                    del iter_likelihood[0]
                    
                    
                for i in range(self.num_ind):
                    var = []
                    for j in range(self.dim):
                        var.append(self.individual[i,j])
                    var.append(self.likelihood[i])
                    self.hist_data.append(var)
                if len(self.hist_data)>int(1e5):
                    self.write_data()
                    self.hist_data       = []
            self.write_data()

            if conv:
                self.iter_likelihood_mean = []
                self.iter_likelihood_best = []
                self.iter_likelihood_median = []
                
                #Change function to modified version
                mod.prob_func = 'mod_' + self.problem_func
                self.best   = mod.abs_best
                best        = mod.abs_best
                self.num_ind = self.num_ind*100
                #Initialize population anew
                self.initialize_population(self.xmin, self.xmax, self.num_ind)
                mod.__init__(self.individual, self.likelihood, mod.prob_func)
                mod.Data.param_change(best=best, delta_log = self.log_thresh)
                
                #Start the evolution
                while self.nfe <= (self.max_nfe):
                    mod.evolve_explore()
                    self.check_oob()
                    self.nfe  += self.num_ind

                    for i in range(self.num_ind):
                        var = []
                        for j in range(self.dim):
                            var.append(self.individual[i,j])
                        var.append(self.likelihood[i])
                        var.append([1])
                        self.hist_data.append(var)

                    if len(self.hist_data)>int(1e5):
                        self.write_data()
                        self.hist_data       = []
                    self.iter_likelihood_mean.append(np.mean(mod.true_likelihood))
                    self.iter_likelihood_best.append(np.min(mod.true_likelihood))
                    self.iter_likelihood_median.append(np.median(mod.true_likelihood))
            self.write_data()

        if self.method == 'double_shade_pso':
            
            iter_likelihood = []; tol = 1e-3; conv = False
            mod = d_SHADE_pso(self.individual, self.likelihood, self.problem_func)
            
            b = 1
            
            while self.nfe < self.max_nfe and conv == False:
                mod.evolve_converge()
                self.check_oob()
                
                self.nfe  += mod.nfe
                iter_likelihood.append(np.mean(mod.likelihood))
                if mod.abs_best < self.best:
                    self.best = mod.abs_best
                if mod.abs_best <tol:
                    print('Converged at nfe:', self.nfe)    
                    print()
                    for i in range(self.num_ind):
                        var = []
                        for j in range(self.dim):
                            var.append(self.individual[i,j])
                        var.append(self.likelihood[i])
                        var.append(1)
                        self.hist_data.append(var)
                    conv = True
                    conv_iter = self.nfe
                    print()
                    print()
                
                if int(b*1e4)<self.nfe:
                    print('nfe:',self.nfe)
                    b+=1
                    
                # for i in range(self.num_ind):
                #     var = []
                #     for j in range(self.dim):
                #         var.append(self.individual[i,j])
                #     var.append(self.likelihood[i])
                #     var.append(1)
                #     self.hist_data.append(var)
                
                if len(mod.hist_data)>int(1e5):
                    self.write_data(mod.hist_data)
                    mod.hist_data = []
            print('Best value:', mod.abs_best)
            print('In: ', mod.best_ind)
            self.write_data(mod.hist_data)
            mod.hist_data = []
            if conv:
                self.iter_likelihood_mean = []
                self.iter_likelihood_best = []
                self.iter_likelihood_median = []
                
                #Change function to modified version
                mod.prob_func = 'mod_' + self.problem_func
                self.best   = mod.abs_best
                best        = mod.abs_best
                self.num_ind = self.num_ind*10
                print('Starting Exploration, NFE:', self.nfe)
                print()
                #Initialize population anew                
                self.initialize_population(self.xmin, self.xmax, self.num_ind)
                mod.__init__(self.individual, self.likelihood, mod.prob_func)
                mod.Data.param_change(best=self.best, delta_log = self.log_thresh)

                #Start the evolution
                for _ in range(self.nfe, int(self.max_nfe), self.num_ind):
                    mod.evolve_explore()
                    self.check_oob()
                    self.nfe  += mod.nfe
                    if int(b*1e4)<self.nfe:
                        print('nfe:',self.nfe)
                        b+=1

                    # for i in range(self.num_ind):
                    #     var = []
                    #     for j in range(self.dim):
                    #         var.append(self.individual[i,j])
                    #     var.append(mod.true_likelihood[i])
                    #     var.append(2)
                    #     self.hist_data.append(var)

                
                    if len(mod.hist_data)>int(1e5):
                        self.write_data(mod.hist_data)
                        mod.hist_data = []
                    # self.iter_likelihood_mean.append(np.mean(mod.true_likelihood))
                    # self.iter_likelihood_best.append(np.min(mod.true_likelihood))
                    # self.iter_likelihood_median.append(np.median(mod.true_likelihood))
                    if self.nfe/self.max_nfe >= 0.6:# or self.nfe >conv_iter + int(conv_iter/2):
                        print('Partikkelisering. NFE: ', self.nfe)
                        print()
                        print('Kriterier:',self.nfe/self.max_nfe, 'og', conv_iter + 3*int(conv_iter/2))
                        break
                self.write_data(mod.hist_data)
                mod.hist_data = []
                #Anders mener k = 5 er nok
                centroids, clusters = mod.cluster(loglike_tol = 5.915, k = 25)

                mod.Data.param_change(best=self.best, delta_log = self.log_thresh)

                mod.optimum = mod.abs_best
                
                
                
                for i in range(self.num_ind):
                    arg = int(clusters[i])
                    mod.optimal_individual[i] = centroids[arg]
                    #for j in range(self.dim):
                     #   arg = clusters[i,j]
                        # arg = mod.un_empty_clusters
                        
                      #  mod.optimal_individual[i,j] = centroids[arg, j]

                mod.init_particle()
                
                for _ in range(self.nfe, int(self.max_nfe), self.num_ind):
                    mod.evolve_particle()
                    self.check_oob()
                    self.nfe  += mod.nfe

                    if int(b*1e4)<self.nfe:
                        print('nfe:',self.nfe)
                        b+=1                    
                    # for i in range(self.num_ind):
                    #     var = []
                    #     for j in range(self.dim):
                    #         var.append(self.individual[i,j])
                    #     var.append(mod.true_likelihood[i])
                    #     var.append(3)
                    #     self.hist_data.append(var)

                    if len(mod.hist_data)>int(1e5):
                        self.write_data(mod.hist_data)
                        mod.hist_data = []
                self.write_data(mod.hist_data)
                mod.hist_data = []

        if self.method == 'double_shade_bat':
            
            iter_likelihood = []; tol = 1e-3; conv = False
            mod = d_SHADE_bat(self.individual, self.likelihood, self.problem_func)
            
            b = 1
            
            while self.nfe < self.max_nfe and conv == False:
                mod.evolve_converge()
                self.check_oob()
                
                self.nfe  += self.num_ind
                iter_likelihood.append(np.mean(mod.likelihood))
                if mod.abs_best < self.best:
                    self.best = mod.abs_best
                if int(self.nfe/self.num_ind) > 10:
                    if mod.abs_best <tol:
                        print('Converged at nfe:', self.nfe)    
                        print()
                        for i in range(self.num_ind):
                            var = []
                            for j in range(self.dim):
                                var.append(self.individual[i,j])
                            var.append(self.likelihood[i])
                            var.append(1)
                            self.hist_data.append(var)
                        conv = True
                        conv_iter = self.nfe
                        print()
                        print()
                    del iter_likelihood[0]
                
                if int(b*1e4)<self.nfe:
                    print('nfe:',self.nfe)
                    b+=1
                    
                for i in range(self.num_ind):
                    var = []
                    for j in range(self.dim):
                        var.append(self.individual[i,j])
                    var.append(self.likelihood[i])
                    var.append(1)
                    self.hist_data.append(var)
                
                if len(mod.hist_data)>int(1e5):
                    self.write_data(mod.hist_data)
                    mod.hist_data = []
            self.write_data(mod.hist_data)
            mod.hist_data = []
            print('Best value:', mod.abs_best)
            print('In: ', mod.best_ind)
            if conv:
                self.iter_likelihood_mean = []
                self.iter_likelihood_best = []
                self.iter_likelihood_median = []
                
                #Change function to modified version
                mod.prob_func = 'mod_' + self.problem_func
                self.best   = mod.abs_best
                best        = mod.abs_best
                self.num_ind = self.num_ind*10
                print('Starting Exploration, NFE:', self.nfe)
                print()
                #Initialize population anew                
                self.initialize_population(self.xmin, self.xmax, self.num_ind)
                mod.__init__(self.individual, self.likelihood, mod.prob_func)
                mod.Data.param_change(best=self.best, delta_log = self.log_thresh)
                #Start the evolution
                for _ in range(self.nfe, int(self.max_nfe), self.num_ind):
                    # print(mod.likelihood)
                    mod.evolve_explore()
                    self.check_oob()
                    self.nfe  += self.num_ind
                    if int(b*1e4)<self.nfe:
                        print('nfe:',self.nfe)
                        b+=1

                    for i in range(self.num_ind):
                        var = []
                        for j in range(self.dim):
                            var.append(self.individual[i,j])
                        var.append(mod.true_likelihood[i])
                        var.append(1)
                        self.hist_data.append(var)

                    if len(mod.hist_data)>int(1e5):
                        self.write_data(mod.hist_data)
                        mod.hist_data = []
                    self.write_data(mod.hist_data)
                    mod.hist_data = []
                    if self.nfe/self.max_nfe >= 0.6:# or self.nfe >conv_iter + int(conv_iter/2):
                        print('Partikkelisering. NFE: ', self.nfe)
                        print()
                        print('Kriterier:',self.nfe/self.max_nfe, 'og', conv_iter + 3*int(conv_iter/2))
                        break
                
                #Anders mener k = 5 er nok per dimensjon
                centroids, clusters = mod.cluster(loglike_tol = 5.915, k = 25)
                mod.Data.param_change(best=self.best, delta_log = self.log_thresh)

                mod.optimum = mod.abs_best
                
                
                
                for i in range(self.num_ind):
                    arg = int(clusters[i])
                    mod.optimal_individual[i] = centroids[arg]
                    #for j in range(self.dim):
                     #   arg = clusters[i,j]
                        # arg = mod.un_empty_clusters
                        
                      #  mod.optimal_individual[i,j] = centroids[arg, j]

                # print(mod.optimal_individual)                
                mod.init_bat()
                mod.set_limits(self.xmin,self.xmax)
                for _ in range(self.nfe, int(self.max_nfe), self.num_ind):
                    if int(b*1e4)<self.nfe:
                        print('nfe:',self.nfe)
                        b+=1                    
                    mod.evolve_bat()
                    self.check_oob()
                    self.nfe  += self.num_ind
                    for i in range(self.num_ind):
                        var = []
                        for j in range(self.dim):
                            var.append(self.individual[i,j])
                        var.append(mod.true_likelihood[i])
                        var.append(2)
                        self.hist_data.append(var)

                    if len(mod.hist_data)>int(1e5):
                        self.write_data(mod.hist_data)
                        mod.hist_data = []
                self.write_data(mod.hist_data)
                mod.hist_data = []

        if self.method == 'shade_bat':
            iter_likelihood = []; tol = 1e-3; conv = False
            mod = shade_bat(self.individual, self.likelihood, self.problem_func)

            while self.nfe < self.max_nfe and conv == False:
                mod.evolve_converge()
                self.check_oob()
                
                self.nfe  += self.num_ind
                iter_likelihood.append(np.mean(mod.true_likelihood))
                
                if int(self.nfe/self.num_ind) > 10:
                    if (np.mean(iter_likelihood)-iter_likelihood[-1]) < tol or any(self.likelihood) == 0:
                        print('Conv')    
                        
                        conv = True
                    del iter_likelihood[0]
                    
                    
                for i in range(self.num_ind):
                    var = []
                    for j in range(self.dim):
                        var.append(self.individual[i,j])
                    var.append(self.likelihood[i])
                    self.hist_data.append(var)
                if len(self.hist_data)>int(1e5):
                    self.write_data()
                    self.hist_data       = []
            self.write_data()

            if conv:

                #Change function to modified version
                mod.prob_func = 'mod_' + self.problem_func
                self.best   = mod.abs_best
                best        = mod.abs_best
                
                #Here we need to calculate the delta_log, ie.  the boundary of likelihood and how many sigma we are interested in.
                #delta_log = 
                #HB:
                #1sig :: 1.15
                #2sig :: 3.09
                #3sig :: 5.915
                
                #Initialize population anew
                self.num_ind = self.num_ind*100
                self.initialize_population(self.xmin, self.xmax, self.num_ind)
                mod.__init__(self.individual, self.likelihood, mod.prob_func)
                mod.Data.param_change(best=best, delta_log = 1.15)
                mod.initialize_bat(True)
                self.bat_memory_individual = mod.BM
                self.bat_memory_likelihood = mod.BM_val
                
                mod.set_limits(self.xmin, self.xmax)
                #Start the evolution
                
                while self.nfe < self.max_nfe:
                    mod.evolve_bat()
                    self.nfe  += self.num_ind
                    self.bat_memory_individual = mod.BM
                    self.bat_memory_likelihood = mod.BM_val

                    for i in range(self.num_ind):
                        var = []
                        for j in range(self.dim):
                            var.append(self.individual[i,j])
                        var.append(self.likelihood[i])
                        self.hist_data.append(var)
                    if len(self.hist_data)>int(1e5):
                        self.write_data()
                        self.hist_data       = []
                self.write_data()
            
    #Sjekker rammebetingelser
    def check_oob(self):
        var = 0
        Data    = Problem_Function(self.dim)
        # Data.param_change(0, 1.15)
        for i in range(self.num_ind):
            for j in range(self.dim):
                if  self.individual[i,j] < self.xmin:
                    var = self.xmin - (self.individual[i,j] - self.xmin)
                    if var > self.xmax or var < self.xmin:
                        self.individual[i,j] = np.random.uniform(self.xmin,self.xmax)
                        temp, z = Data.evaluate(self.individual[i,:], self.problem_func)
                        self.likelihood[i] = temp
                    else:
                        self.individual[i,j] = var
                        temp, z = Data.evaluate(self.individual[i,:], self.problem_func)
                        self.likelihood[i] = temp

                if  self.individual[i,j] > self.xmax:
                    var  = self.xmax - (self.individual[i,j] - self.xmax)
                    if var < self.xmin or var > self.xmax:
                        self.individual[i,j] = np.random.uniform(self.xmin,self.xmax)
                        temp, z = Data.evaluate(self.individual[i,:], self.problem_func)
                        self.likelihood[i] = temp
                        
                    else:
                        self.individual[i,j] = var
                        temp, z = Data.evaluate(self.individual[i,:], self.problem_func)
                        self.likelihood[i] =  temp
    
    
    def open_data(self):
        # self.path = (r'C:\Users\Bendik Selvaag-Hagen\OneDrive - Universitetet i Oslo\Documents\GitHub\Master\datafile.csv')
        self.path = (r"C:\Users\Lenovo\Documents\Master\datafile.csv")
        with open(self.path, 'w', newline='') as csvfile:
            csvfile = csv.writer(csvfile, delimiter=',')
            csvfile.writerows(self.hist_data)
            
    # def write_data(self):
    #     # print('Length of written datafile:',len(self.hist_data))
    #     with open(self.path, 'a', newline='') as csvfile:
    #         csvfile = csv.writer(csvfile, delimiter=',')
    #         csvfile.writerows(self.hist_data)
    #     self.hist_data       = []
    def write_data(self, data):
        # print('Length of written datafile:',len(self.hist_data))
        with open(self.path, 'a', newline='') as csvfile:
            csvfile = csv.writer(csvfile, delimiter=',')
            csvfile.writerows(data)


        
 
#¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
#Available methods:
# ¤ jDE /jDErpo
# ¤ bat
# ¤ shade / double_shade / shade_bat / double_shade_pso
#¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
np.random.seed(4531232)

def log_thresh(sigma):
    if sigma == 1:
        return 1.15
    if sigma == 2:
        return 3.09
    if sigma == 3:
        return 5.915

dim                 = 5
population_size     = 100
max_nfe             = 1e6  
method              = 'double_shade_pso'
# method              = 'jderpo'
# problem_function    = 'Himmelblau'
problem_function    = 'Rosenbrock'
# problem_function    = 'Eggholder'
xmin, xmax = conditions(problem_function)

#How many sigmas are you interested in?
sigma = 2
log_threshold = log_thresh(sigma)


cl = DEVO_class(dim, problem_function, method, log_threshold)
cl.initialize_population(xmin,xmax, population_size)
cl.evolve(max_nfe)
print('Program Complete. Analyzing data and Plotting.')


vis = Vis(dim, xmin,xmax, max_nfe, method, problem_function)
vis.extract_data()
# vis.visualize_parameter_space()
vis.stacked_hist()


#h5py
#text-behandling _



# SKaler hastighet partikkel med størrelse clsuter
