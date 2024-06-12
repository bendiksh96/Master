import sys
import numpy as np
import random as rd
import csv
import time
sys.path.append(r'C:\Users\Lenovo\Documents\GitHub\Master\Methods')
# sys.path.append(r'C:\Users\Lenovo\Documents\Master\Methods')
# sys.path.append(r'C:\Users\Bendik Selvaag-Hagen\OneDrive - Universitetet i Oslo\Documents\GitHub\Master')

from vis_writ import *
# sys.path.append(r'C:\Users\Bendik Selvaag-Hagen\OneDrive - Universitetet i Oslo\Documents\GitHub\Master\Methods')
# sys.path.append(r'C:\Users\Bendik Selvaag-Hagen\Desktop\Skole\Master\Methods')
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


def conditions(problem_function):
    if problem_function == 'Himmelblau':
        xmin, xmax = -5,5
    if problem_function == 'Eggholder':
        xmin, xmax = -512,512
    if problem_function == 'Rosenbrock':
        xmin, xmax = -5,10
    if problem_function == 'Michalewicz':
        xmin, xmax = 0, np.pi
    if problem_function == 'Rotated_Hyper_Ellipsoid':
        xmin, xmax = -50,50
    if problem_function == 'Hartman_3D':
        xmin, xmax = 0,1
    if problem_function == 'Levy':
        xmin, xmax = -10, 10
    if problem_function == 'Rastrigin':
        xmin, xmax = -5,5 
    if problem_function == 'Ackley':
        xmin, xmax = -32, 32
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
        self.nu_pop         = 0
        self.init_pso       = 0.8
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
        for i in range(self.num_ind):
            var     = []
            for j in range(self.dim):
                self.individual[i,j] = np.random.uniform(xmin, xmax)
                var.append(self.individual[i,j])
            temp, z = Data.evaluate(self.individual[i,:], self.problem_func)
            self.likelihood[i] = temp
            var.append(z)
            var.append(1)
            self.hist_data.append(var)
        self.nfe += self.num_ind

        self.write_data(self.hist_data)
        self.hist_data = []

    def evolve(self, max_nfe):
        self.max_nfe   = max_nfe
        if self.method == 'random_search':
            Data    = Problem_Function(self.dim)

            while self.nfe < self.max_nfe:
                for i in range(self.num_ind):
                    var = []
                    for j in range(self.dim):
                        self.individual[i,j] = np.random.uniform(self.xmin, self.xmax)
                        var.append(self.individual[i,j])
                    temp, z = Data.evaluate(self.individual[i,:], self.problem_func)
                    self.likelihood[i] = z
                    var.append(temp)
                    var.append(1)

                    self.hist_data.append(var)
                self.nfe += self.num_ind
                if len(self.hist_data)>int(1e5):
                    self.write_data(self.hist_data)
                    self.hist_data       = []


            self.write_data(self.hist_data)

        if self.method == 'jde':
            b = 1
            iter_likelihood = []; tol = 1e-3; conv = False
            mod = jDE(self.individual, self.likelihood, self.problem_func, self.xmin, self.xmax)
            while self.nfe < self.max_nfe and conv == False:
                mod.evolve()
                self.nfe  += mod.nfe
                if int(b*1e4)<self.nfe:
                    print('nfe:',self.nfe)
                    b+=1

                if len(self.hist_data)>int(1e5):
                    self.write_data(mod.hist_data)
                    mod.hist_data       = []

                if len(self.hist_data)>int(1e5):
                    self.write_data(mod.hist_data)
                    mod.hist_data       = []
            self.write_data(mod.hist_data)

        if self.method == 'jderpo':
            iter_likelihood = []; tol = 1e-3; conv = False
            mod = jDErpo(self.individual, self.likelihood, self.problem_func, self.xmin, self.xmax)
            mod.max_nfe = self.max_nfe
            b = 1
            while self.nfe < self.max_nfe and conv == False:
                mod.evolve()
                #self.check_oob()
                self.nfe  += mod.nfe
                iter_likelihood.append(np.mean(mod.likelihood))
                if int(b*1e4)<self.nfe:
                    print('nfe:',self.nfe)
                    b+=1

                if len(mod.hist_data)>int(1e5):
                    self.write_data(mod.hist_data)
                    mod.hist_data       = []
            self.write_data(mod.hist_data)

        if self.method == 'shade':
            b = 1
            iter_likelihood = []; tol = 1e-3; conv = False
            mod = SHADE(self.individual, self.likelihood, self.problem_func, self.xmin, self.xmax)
            while self.nfe < self.max_nfe and conv == False:
                #Evolve the algorithm
                mod.evolve()
                #self.check_oob()
                self.nfe  += mod.nfe

                if int(b*1e4)<self.nfe:
                    print('nfe:',self.nfe)
                    b+=1

                #If history too large, write to file and reset
                if len(mod.hist_data)>int(1e5):
                    self.write_data(mod.hist_data)
                    mod.hist_data = []
            self.write_data(mod.hist_data)

        if self.method == 'double_shade':
            iter_likelihood = []; tol = 1e-3; conv = False
            mod = d_SHADE(self.individual, self.likelihood, self.problem_func, self.xmin, self.xmax)
            b = 1

            while self.nfe < self.max_nfe and conv == False:
                mod.evolve_converge()
                self.nfe  += mod.nfe
                if mod.abs_best < self.best:
                    self.best = mod.abs_best
                if mod.abs_best <tol:
                    print('Converged at nfe:', self.nfe)
                    print()

                    conv = True
                    conv_iter = self.nfe
                    print()
                    print()

                if int(b*1e4)<self.nfe:
                    print('nfe:',self.nfe)
                    b+=1

                if len(mod.hist_data)>int(3e4):
                    self.write_data(mod.hist_data)
                    mod.hist_data = []

            self.write_data(mod.hist_data)
            print(len(mod.hist_data))
            """
            mod.hist_data = []
            print('Best value:', mod.abs_best)
            print('In: ', mod.best_ind)
            if conv:
                #Change function to modified version
                mod.prob_func = 'mod_' + self.problem_func
                self.best   = mod.abs_best

                self.num_ind = self.num_ind*100
                print('Starting Exploration, NFE:', self.nfe)
                print()
                self.nu_pop = self.num_ind

                #Initialize population anew
                self.initialize_population(self.xmin, self.xmax, self.num_ind)
                mod.__init__(self.individual, self.likelihood, mod.prob_func, self.xmin, self.xmax)
                mod.Data.param_change(best=self.best, delta_log = self.log_thresh)

                #Start the evolution
                while self.nfe < self.max_nfe:
                    mod.evolve_explore()
                    # self.check_oob()
                    self.nfe  += mod.nfe
                    if int(b*1e4)<self.nfe:
                        print('nfe:',self.nfe)
                        b+=1
                    if len(mod.hist_data)>int(1e5):
                        self.write_data(mod.hist_data)
                        mod.hist_data = []
                self.write_data(mod.hist_data)
                mod.hist_data = []
                """
        if self.method == 'double_shade_pso':
            iter_likelihood = []; tol = 1e-3; conv = False
            mod = d_SHADE_pso(self.individual, self.likelihood, self.problem_func, self.xmin, self.xmax)
            b = 1
            # print(self.likelihood)
            while self.nfe < self.max_nfe and conv == False:
                mod.evolve_converge()
                self.nfe  += mod.nfe
                iter_likelihood.append(np.mean(mod.likelihood))
                if mod.abs_best < self.best:
                    self.best = mod.abs_best
                if mod.abs_best <tol:
                    print('Converged at nfe:', self.nfe)
                    conv = True
                    conv_iter = self.nfe
                    print()
                    print()

                if int(b*1e4)<self.nfe:
                    print('nfe:',self.nfe)
                    b+=1

                if len(mod.hist_data)>int(1e5):
                    self.write_data(mod.hist_data)
                    mod.hist_data = []
            print('Best value:', mod.abs_best)
            print('In: ', mod.best_ind)
            self.write_data(mod.hist_data)
            mod.hist_data = []
            if conv:
                #Change function to modified version
                mod.prob_func = 'mod_' + self.problem_func
                self.best   = mod.abs_best
                best        = mod.abs_best

                self.num_ind = self.num_ind*100
                print('Starting Exploration, NFE:', self.nfe)
                print()
                #Initialize population anew
                self.initialize_population(self.xmin, self.xmax, self.num_ind)
                mod.__init__(self.individual, self.likelihood, mod.prob_func, self.xmin, self.xmax)
                mod.Data.param_change(best=self.best, delta_log = self.log_thresh)
                self.nu_pop = self.num_ind
                #Start the evolution
                while self.nfe < self.max_nfe:
                    mod.evolve_explore()
                    # self.check_oob()
                    self.nfe  += mod.nfe
                    if int(b*1e4)<self.nfe:
                        print('nfe:',self.nfe)
                        b+=1

                    if len(mod.hist_data)>int(1e5):
                        self.write_data(mod.hist_data)
                        mod.hist_data = []
                    if self.nfe/self.max_nfe >= self.init_pso:
                        print('Partikkelisering. NFE: ', self.nfe)
                        print()
                        print('Kriterier:',self.nfe/self.max_nfe, 'og', conv_iter + 3*int(conv_iter/2))
                        break

                self.write_data(mod.hist_data)
                mod.hist_data = []

                loglike_tol = 3.2
                k           = 20
                #Anders mener k = 5 er nok per dimensjon
                cluster_check = np.where(self.likelihood < loglike_tol)
                if len(self.likelihood[cluster_check])< k:
                    k = len(self.likelihood[cluster_check])
                    if k == 0:
                        mod.optimal_individual = mod.best_ind
                    else: 
                        centroids, clusters = mod.cluster(loglike_tol, k)
                        mod.Data.param_change(best=self.best, delta_log = self.log_thresh)
                        mod.optimum = mod.abs_best
                        for i in range(self.num_ind):
                            arg = int(clusters[i])
                            mod.optimal_individual[i] = centroids[arg]

                else: 
                    centroids, clusters = mod.cluster(loglike_tol, k)
                    mod.Data.param_change(best=self.best, delta_log = self.log_thresh)
                    mod.optimum = mod.abs_best
                    for i in range(self.num_ind):
                        arg = int(clusters[i])
                        mod.optimal_individual[i] = centroids[arg]


                mod.init_particle()

                for _ in range(self.nfe, int(self.max_nfe), self.num_ind):
                    mod.evolve_particle()
                    # self.check_oob()
                    self.nfe  += mod.nfe
                    if int(b*1e4)<self.nfe:
                        print('nfe:',self.nfe)
                        b+=1
                    if len(mod.hist_data)>int(1e5):
                        self.write_data(mod.hist_data)
                        mod.hist_data = []
                self.write_data(mod.hist_data)
                mod.hist_data = []

        if self.method == 'double_shade_bat':
            iter_likelihood = []; tol = 1e-3; conv = False; b = 1
            mod = d_SHADE_bat(self.individual, self.likelihood, self.problem_func,self.xmin, self.xmax)

            while self.nfe < self.max_nfe and conv == False:
                mod.evolve_converge()
                self.nfe  += mod.nfe

                if mod.abs_best < self.best:
                    self.best = mod.abs_best

                if mod.abs_best <tol:
                    print('Converged at nfe:', self.nfe)
                    print()
                    print()
                    conv = True
                    conv_iter = self.nfe

                if int(b*1e4)<self.nfe:
                    print('nfe:',self.nfe)
                    b+=1

                if len(mod.hist_data)>int(2e4):
                    self.write_data(mod.hist_data)
                    mod.hist_data = []
            self.write_data(mod.hist_data)
            mod.hist_data = []

            print('Best value:', mod.abs_best)
            print('In: ', mod.best_ind)
            print()

            if conv:
                #Change function to modified version
                mod.prob_func = 'mod_' + self.problem_func
                self.best   = mod.abs_best
                best        = mod.abs_best
                self.num_ind = self.num_ind*100
                print('Starting Exploration, NFE:', self.nfe)
                print()
                self.nu_pop = self.num_ind

                #Initialize population anew
                self.initialize_population(self.xmin, self.xmax, self.num_ind)
                mod.__init__(self.individual, self.likelihood, mod.prob_func, self.xmin, self.xmax)
                mod.Data.param_change(best=self.best, delta_log = self.log_thresh)

                #Start the evolution
                while self.nfe < self.max_nfe:
                    mod.evolve_explore()
                    self.nfe  += mod.nfe
                    if int(b*1e4)<self.nfe:
                        print('nfe:',self.nfe)
                        b+=1
                    if len(mod.hist_data)>int(1e5):
                        self.write_data(mod.hist_data)
                        mod.hist_data = []

                    if self.nfe/self.max_nfe >= self.init_pso:
                        print('Flaggermusisering. NFE: ', self.nfe)
                        print()
                        print('Kriterier:',self.nfe/self.max_nfe, 'og', conv_iter + 3*int(conv_iter/2))
                        break


                self.write_data(mod.hist_data)
                mod.hist_data = []
                loglike_tol = 3.2
                k =20
                #Anders mener k = 5 er nok per dimensjon
                cluster_check = np.where(self.likelihood < loglike_tol)
                if len(self.likelihood[cluster_check])< k:
                    k = len(self.likelihood[cluster_check])
                    if k == 0:
                        mod.optimal_individual = mod.best_ind
                    else: 
                        centroids, clusters = mod.cluster(loglike_tol, k)
                        mod.Data.param_change(best=self.best, delta_log = self.log_thresh)
                        mod.optimum = mod.abs_best
                        for i in range(self.num_ind):
                            arg = int(clusters[i])
                            mod.optimal_individual[i] = centroids[arg]
                else: 
                    centroids, clusters = mod.cluster(loglike_tol, k)
                    mod.Data.param_change(best=self.best, delta_log = self.log_thresh)
                    mod.optimum = mod.abs_best
                    for i in range(self.num_ind):
                        arg = int(clusters[i])
                        mod.optimal_individual[i] = centroids[arg]


                mod.init_bat()
                mod.set_limits(self.xmin,self.xmax)
                while self.nfe < self.max_nfe:
                    mod.evolve_bat()
                    self.check_oob()
                    self.nfe  += mod.nfe
                    if int(b*1e4)<self.nfe:
                        print('nfe:',self.nfe)
                        b+=1
                    if len(mod.hist_data)>int(1e5):
                        self.write_data(mod.hist_data)
                        mod.hist_data = []
                self.write_data(mod.hist_data)
                mod.hist_data = []


    # Legg til at oob kandidater må telle opp nfe og legg til nytt individ


    #Sjekker rammebetingelser
    def check_oob(self):
        var = 0
        Data    = Problem_Function(self.dim)
        # Data.param_change(0, 1.15)
        for i in range(self.num_ind):
            for j in range(self.dim):
                if  self.individual[i,j] < self.xmin:
                    lampo = []
                    var = self.xmin - (self.individual[i,j] - self.xmin)
                    if var > self.xmax or var < self.xmin:
                        self.individual[i,j] = np.random.uniform(self.xmin,self.xmax)
                        temp, z = Data.evaluate(self.individual[i,:], self.problem_func)
                        self.likelihood[i] = temp

                        lampo = list(self.individual[i,:])
                        lampo.append(z)
                        self.hist_data.append(lampo)
                        self.nfe += 1

                    else:
                        self.individual[i,j] = var
                        temp, z = Data.evaluate(self.individual[i,:], self.problem_func)
                        self.likelihood[i] = temp

                        self.nfe +=1
                        lampo = list(self.individual[i,:])
                        lampo.append(z)
                        self.hist_data.append(lampo)

                if  self.individual[i,j] > self.xmax:
                    var  = self.xmax - (self.individual[i,j] - self.xmax)
                    if var < self.xmin or var > self.xmax:
                        self.individual[i,j] = np.random.uniform(self.xmin,self.xmax)
                        temp, z = Data.evaluate(self.individual[i,:], self.problem_func)
                        self.likelihood[i] = temp

                        self.nfe +=1
                        lampo = list(self.individual[i,:])
                        lampo.append(z)
                        self.hist_data.append(lampo)


                    else:
                        self.individual[i,j] = var
                        temp, z = Data.evaluate(self.individual[i,:], self.problem_func)
                        self.likelihood[i] =  temp

                        self.nfe +=1
                        lampo = list(self.individual[i,:])
                        lampo.append(z)
                        self.hist_data.append(lampo)


    def open_data(self):
        # self.path = (r'C:\Users\Bendik Selvaag-Hagen\OneDrive - Universitetet i Oslo\Documents\GitHub\Master\datafile.csv')
        self.path = (r"C:\Users\Lenovo\Documents\Master\datafile.csv")
        with open(self.path, 'w', newline='') as csvfile:
            csvfile = csv.writer(csvfile, delimiter=',')
            csvfile.writerows(self.hist_data)

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
seed = 397285674
np.random.seed(seed)

def log_thresh(sigma):
    if sigma == 1:
        return 1.15
    if sigma == 2:
        return 3.09
    if sigma == 3:
        return 5.915

start = time.time()
dim                 = 5
sigma               = 2

def collector():
    method_list =['double_shade']#['jde']# ['double_shade','double_shade_pso','double_shade_pso', 'double_shade_bat']#['jde', 'jderpo', 'shade','random_search','double_shade','double_shade_pso', 'double_shade_bat']
    func_list   = ['Himmelblau', 'Rosenbrock', 'Rastrigin', 'Levy']  
    func_writ   = ['himmelblau', 'rosenbrock', 'rastrigin', 'levy'] 
    if dim == 3:
        nfe_list    = [1e5]#, 2e5, 5e5]
    elif dim == 4: 
        nfe_list    = [1e6]#, 2e6, 5e6]
    elif dim == 5:
        nfe_list    = [1e7]#, 2e7, 5e7]
    else:
        print('Not designed for this dimension')
        exit()
                
    path = (r"C:\Users\Lenovo\Documents\Master\result.csv")
    with open(path, 'w', newline='') as csvfile:
        csvfile = csv.writer(csvfile, delimiter=',')

    for met in range(len(method_list)):
        for fun in range(len(func_list)):
            time_elapse = time.time()
            for nf in range(len(nfe_list)):
                if method_list[met] == 'shade' or method_list[met] == 'jde' or method_list[met] == 'jderpo':
                    population_size = int(nfe_list[nf]/50)
                else:
                    population_size = 243#5**(dim)
                    #Bruker 243 for dshade i 5D, det er 3**5
                xmin, xmax          = conditions(func_list[fun])
                log_threshold       = log_thresh(sigma)
                cl                  = DEVO_class(dim, func_list[fun], method_list[met], log_threshold)
                cl.initialize_population(xmin,xmax, population_size)
                cl.evolve(nfe_list[nf])
                
                if dim == 3:
                    bin_path = "C:/Users/Lenovo/Documents/GitHub/Master/Data/" + func_writ[fun]+"_3D_100_result.npy" 
                if dim == 4:
                    bin_path = "C:/Users/Lenovo/Documents/GitHub/Master/Data/" + func_writ[fun]+"_4D_100_result.npy" 
                if dim == 5:
                    bin_path = "C:/Users/Lenovo/Documents/GitHub/Master/Data/" + func_writ[fun]+"_5D_40_result.npy" 


                dw                  = Vis(dim, xmin,xmax,nfe_list[nf],method_list[met],func_list[fun])
                dw.extract_data()
                dw.bin_parameter_space(bin_path)
                dw.compare_bins(bin_path)
                data = [
                    ['Method:', method_list[met] , 'Function:' , func_list[fun] , 'NFE:', nfe_list[nf]],
                    ['score within threshold', dw.occ],
                    ['score on contour', dw.delta_occ],
                    ['min', dw.mini],
                    ['pop', population_size],
                    ['contour %', dw.bin_count_cont/dw.bin_count_cont_valid],
                    ['below threshold %', dw.bin_count_sigm/dw.bin_count_sigm_valid], '\n']
                
                with open(path, 'a', newline='') as csvfile:
                    w = csv.writer(csvfile,delimiter= ',')
                    w.writerows(data)
            time_durian = time.time()
            print('This took', time_durian-time_elapse, 'seconds')

collector()         
end = time.time()
print('Total elapsed time:', end-start)

"""
            
# max_nfe             = 1e5
# method              = 'random_search'
# method              = 'shade'     ; population_size = int(max_nfe/50)
# method              = 'double_shade'
# method              = 'double_shade_pso'
# method              = 'double_shade_bat'
# method              = 'jde'     ; population_size = int(max_nfe/50)
# method              = 'jderpo'
# method              = 'chaotic_bat'


# problem_function    = 'Himmelblau'
# problem_function    = 'Rosenbrock'
# problem_function    = 'Eggholder'
# xmin, xmax = conditions(problem_function)

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
# vis.stacked_hist()
vis.bin_parameter_space()
vis.compare_bins()
"""
