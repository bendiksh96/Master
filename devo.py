import numpy as np
from problem_func import *


class DEVO:
    def __init__(self, population_size, mut_prob, no_parents, dim, problem_func,  no_pop = 1):
        self.no_pop = no_pop
        self.population_size = population_size
        self.mut_prob = mut_prob
        self.no_parents = no_parents
        self.dim = dim
        self.problem_func = problem_func
        self.abs_best = 1000
        #Need to have an option for different population sizes for different populations.
            
        if no_pop == 1:
            self.ind = np.zeros((self.population_size, self.dim))
            self.likely_ind = np.zeros((self.population_size, 1))
    
    #Trenger å kunne bestemme dimensjonen på popluasjonen og gi dette som et output
    def initialize_single_pop(self,dim, x_min, x_max, randomize = 'True'):
        self.dim = dim
        self.x_min = x_min
        self.x_max = x_max
        if randomize:
            for i in range(self.population_size):
                rand_ = np.zeros(self.dim)
                for j in range(self.dim):
                    rand_[j] = np.random.uniform(x_min, x_max)
                self.ind[i] = rand_
        else:
            pass
    
    def multipop_diff_ind_size(self, dim, x_min, x_max, multipop_ind_list, randomize=True):
        self.dim = dim
        self.x_min = x_min
        self.x_max = x_max
        
        self.ind = []
        self.likely_ind = []
        for i in range(self.no_pop):
            arr = np.array((multipop_ind_list[i], self.dim))
            arr_ = np.array(multipop_ind_list[i])
            self.ind.append(arr)
            self.likely_ind.append(arr_)

    #Mechanism retaining the individuals within the bounds    
    def check_oob(self):
        for i in range(self.population_size):
            for j in range(self.dim):
                if self.ind[i,j] < self.x_min:
                    #Sprettball
                    self.ind[i,j] = self.x_min - (self.ind[i,j] - self.x_min)
            if self.ind[i,j] > self.x_max:
                    #Sprettball
                    self.ind[i,j] = self.x_max - (self.ind[i,j] - self.x_max)
                    
    #Må evalurere iht. testfunksjon
    def eval_likelihood_pop(self):
        func = Problem_Function(self.dim)
        if self.problem_func == "Rosenbrock":
            for i in range(self.population_size):
                score = func.Rosenbrock(self.ind[i])
                self.likely_ind[i] = score

                
        elif self.problem_func == "Eggholder":
            for i in range(self.population_size):
                score = func.Eggholder(self.ind[i])
                self.likely_ind[i] = score

    def eval_likelihood_ind(self, ind):
        func = Problem_Function(self.dim)
        if self.problem_func == "Rosenbrock":
            score = func.Rosenbrock(ind)
        elif self.problem_func == "Eggholder":
            score = func.Eggholder(ind)
        return score
    
    #Velg foreldre iht. antall foreldre, samt likelihood
    def fornicate(self, method='Binomial Crossover'):
        #De beste n verdiene
        if method == 'Binomial Crossover':
            sort = np.argsort(self.likely_ind[:],axis=0)
            rank_likelihood = self.likely_ind[sort]
            rank_ind = self.ind[sort]
            if rank_likelihood[0] < self.abs_best:
                self.abs_best = rank_likelihood[0]
                self.best_ind = rank_ind[0]        
            self.offspring = np.zeros((int(self.no_parents/2), self.dim))
            for i in range(0, int(self.no_parents), 2):
                offspring1 = []
                for j in range(self.dim):
                    randint = np.random.randint(0,1)
                    if randint == 1:
                        offspring1.append(rank_ind[i,0,j])
                    else:
                        offspring1.append(rank_ind[i+1,0,j])
                self.offspring[int(i/2)] = offspring1
            self.mut_amp = 0.4
            for i in range(self.population_size):
                for j in range(self.dim):
                    if (self.mut_prob>np.random.uniform(0,1)):
                        rand_1 = np.random.randint(0,self.population_size)
                        rand_2 = np.random.randint(0,self.population_size)
                        self.ind[i,j] = self.ind[i,j] + self.mut_amp*(self.ind[rand_1,j]-self.ind[rand_2,j])

            #Turnering
            for i in range(int(self.no_parents/2)):
                randint = np.random.randint(0,self.population_size)
                off_value = self.eval_likelihood_ind(self.offspring[i])
                if off_value<self.likely_ind[randint]:
                    self.ind[randint] = self.offspring[i]
                    self.likely_ind[randint] = off_value
        if method == 'jDE':
            F = 0.1
            CR = 0.1
            
            self.u = np.zeros_like(self.ind)
            self.v = np.zeros_like(self.ind)
            klai = np.argsort(self.likely_ind, axis = 0)
            best_index = klai[0]
            # print(best_index)
            ind_best = self.ind[best_index]
            #Mutant vector
            for i in range(self.population_size):
                rand1_ = np.random.randint(self.population_size)
                rand2_ = np.random.randint(self.population_size)
                rand3_ = np.random.randint(self.population_size)
                
                #rand/1 scheme
                self.v[i] = self.ind[rand1_] + F * (self.ind[rand2_]- self.ind[rand3_])
            
                #best/1 scheme
                #Denne suger
                # self.v[i] = ind_best + F * (self.ind[rand2_] - self.ind[rand3_])
            
            for i in range(self.population_size):
                #Crossover
                for j in range(self.dim):
                    randint = np.random.randint(0,1)
                    if randint < CR:
                        self.u[i,j] = self.v[i,j]
                #Procreate
                if self.eval_likelihood_ind(self.u[i]) < self.likely_ind[i]:
                    self.ind[i] = self.u[i]

    #Ranger barn opp mot foreldre og populasjon. Velg hvilke som får bli. 
    def select_offspring(self):
        #Velger random individ og erstatter med barn
        # for i in range(self.no_parents):
        #     randint = np.random.randint(0,self.population_size)
        #     self.ind[randint] = self.offspring[i]
        
        #Turnering
        for i in range(int(self.no_parents/2)):
            randint = np.random.randint(0,self.population_size)
            off_value = self.eval_likelihood_ind(self.offspring[i])
            if off_value<self.likely_ind[randint]:
                self.ind[randint] = self.offspring[i]
                self.likely_ind[randint] = off_value
        
    #What kind of evoltionary mechanism. Need to afflict the parent selection and likelihood. 
    def evol_type(self, evol = 'darwin'):
        self.d = 1
        if evol == 'bald':
            pass
        elif evol == 'lama':
            pass
        
    #Need to 
    def mutation(self):
        self.mut_amp = 0.4
        for i in range(self.population_size):
            for j in range(self.dim):
                if (self.mut_prob>np.random.uniform(0,1)):
                    rand_1 = np.random.randint(0,self.population_size)
                    rand_2 = np.random.randint(0,self.population_size)
                    self.ind[i,j] = self.ind[i,j] + self.mut_amp*(self.ind[rand_1,j]-self.ind[rand_2,j])
    
    