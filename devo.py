import numpy as np
import random as rd
from problem_func import *
from scipy.stats import cauchy

class DEVO:
    def __init__(self, maxiteration,population_size_list, mut_prob, no_parents, dim, problem_func,  no_pop = 1, prior_best=0):
        self.iter = 0
        self.maxiter = maxiteration
        self.no_pop = no_pop
        self.conv = False
        self.population_size_list = population_size_list
        # self.population_size = population_size
        self.mut_prob_list = mut_prob
        self.no_parents = no_parents
        self.dim = dim
        self.problem_func = problem_func
        self.abs_best = 1000
        #Need to have an option for different population sizes for different populations.
        self.ind = []
        self.likely_ind = []
        self.likely_ind_percieved = []
        self.F_list = []
        self.CR_list = []
        self.tau1   = 0.1
        self.tau2   = 0.1
        self.CRupp  = 1
        self.Fupp   = 1
        self.A = []
        
        self.A_all = [] ; self.A_re = [] ; self.A_ll = []
        
        
        self.k_arg = 1

        self.std = 0
        self.M_CR = []
        self.M_F  = []
        
        for pop in range(self.no_pop):
            a = np.ones((self.population_size_list[pop]))
            self.ind.append(np.zeros((self.population_size_list[pop], self.dim)))
            self.likely_ind.append(np.zeros((self.population_size_list[pop])))
            self.likely_ind_percieved.append(np.zeros((self.population_size_list[pop])))
            self.F_list.append(a)
            self.CR_list.append(a)
            self.M_CR.append(0.5*a)
            self.M_F.append(0.5*a)
            
        self.func_calls = 0
        
        #For second run
        self.prior_best = prior_best
        self.sigma      = 0.5
        self.delta      = 200

    #Trenger å kunne bestemme dimensjonen på popluasjonen og gi dette som et output
    def initialize_single_pop(self, x_min, x_max, randomize = 'True'):
        self.x_min = x_min
        self.x_max = x_max
        for pop in range(self.no_pop):
            if randomize:
                for i in range(self.population_size_list[pop]):
                    for j in range(self.dim):
                        rand_ = np.random.uniform(x_min, x_max)
                        self.ind[pop][i,j] = rand_
                # self.std = np.std(self.likely_ind[pop][:])
            if randomize == False:
                for i in range(self.population_size_list[pop]):
                    for j in range(self.dim):
                        randy = np.random.randint(-1,1)
                        if randy:
                            self.ind[pop][i,j] = np.random.uniform(x_min, x_max)
                        elif randy == 0:
                            self.ind[pop][i,j] = np.random.normal(x_min, x_max)
                        elif randy == -1:
                            self.ind[pop][i,j] = 1- np.random.normal(x_min, x_max)
            if pop ==1:

                available_indices = [set(range(self.population_size_list[pop])) for a in range(self.dim)]
                samples = []
                for i in range(self.population_size_list[pop]//2):
                    sample1 = list()
                    sample2 = list()

                    for idx in available_indices:
                        k = rd.sample(idx, 1)[0]
                        sample1.append(k)
                        sample2.append(self.population_size_list[pop]-1-k)
                        idx.remove(k)
                        idx.remove(self.population_size_list[pop]-1-k)

                    samples.append(sample1)
                    samples.append(sample2)

                samples = np.array(samples)
                self.A_ll.append(self.ind[pop])
                
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
        for pop in range(self.no_pop):
            for i in range(self.population_size_list[pop]):
                for j in range(self.dim):
                    if self.ind[pop][i,j] < self.x_min:
                        #Sprettball
                        self.ind[pop][i,j] = self.x_min - (self.ind[pop][i,j] - self.x_min)
                    if self.ind[pop][i,j] > self.x_max:
                        #Sprettball
                        self.ind[pop][i,j] = self.x_max - (self.ind[pop][i,j] - self.x_max)
                        
    #Evaluer en populasjons likelihood
    def eval_likelihood_pop(self):
        func = Problem_Function(self.dim)
        if self.problem_func == "Rosenbrock":
            for pop in range(self.no_pop):
                for i in range(self.population_size_list[pop]):
                    score = func.Rosenbrock(self.ind[pop][i])
                    self.likely_ind[pop][i]= score
                    self.func_calls += 1
        elif self.problem_func == "Eggholder":
            for pop in range(self.no_pop):
                for i in range(self.population_size_list[pop]):
                    score = func.Eggholder(self.ind[pop][i])
                    self.likely_ind[pop][i] = score
                    self.func_calls += 1

        elif self.problem_func == "mod_eggholder":
            for pop in range(self.no_pop):
                for i in range(self.population_size_list[pop]):
                    score = func.mod_eggholder(self.ind[pop][i], self.prior_best, self.delta, self.sigma)
                    self.likely_ind[pop][i] = score
                    self.func_calls += 1


        elif self.problem_func == "Himmelblau":
            for pop in range(self.no_pop):
                for i in range(self.population_size_list[pop]):
                    score = func.Himmelblau(self.ind[pop][i])
                    self.likely_ind[pop][i] = score
                    self.func_calls += 1

    #Evaluer et individs likelihood
    def eval_likelihood_ind(self, ind):
        func = Problem_Function(self.dim)
        if self.problem_func == "Rosenbrock":
            score = func.Rosenbrock(ind)
            self.func_calls += 1

        elif self.problem_func == "Eggholder":
            score = func.Eggholder(ind)
            self.func_calls += 1

        elif self.problem_func == "mod_eggholder":
            score = func.mod_eggholder(ind, self.prior_best, self.delta, self.sigma)
            self.func_calls += 1


        elif self.problem_func == "Himmelblau":
            score = func.Himmelblau(ind)
            self.func_calls += 1

        return score
    
    #Velg foreldre iht. antall foreldre, samt likelihood
    def fornicate(self, method='Binomial Crossover'):
        #De beste n verdiene
        if method == 'Binomial Crossover':
            for pop in range(self.no_pop):
                sort = np.argsort(self.likely_ind_percieved[pop][:],axis=0)
                rank_likelihood = self.likely_ind_percieved[pop][sort]
                # sort = np.argsort(self.likely_ind[pop,:],axis=0)
                # rank_likelihood = self.likely_ind[pop,sort]
                rank_ind = self.ind[pop][sort]
                # if rank_likelihood[pop][0] < self.abs_best:
                #     self.abs_best = rank_likelihood[pop][0]
                #     self.best_ind = rank_ind[pop][0]        
                self.offspring = np.zeros((int(self.no_parents[pop]/2), self.dim))
                for i in range(0, int(self.no_parents[pop]/2), 2):
                    offspring1 = []
                    for j in range(self.dim):
                        randint = np.random.randint(0,1)
                        if randint == 1:
                            offspring1.append(rank_ind[i,j])
                        else:
                            offspring1.append(rank_ind[i+1,j])
                    self.offspring[int(i/2)] = offspring1
                self.mut_amp = 0.4
                for i in range(self.population_size_list[pop]):
                    for j in range(self.dim):
                        if (self.mut_prob_list[pop]>np.random.uniform(0,1)):
                            rand_1 = np.random.randint(0,self.population_size_list[pop])
                            rand_2 = np.random.randint(0,self.population_size_list[pop])
                            self.ind[pop][i,j] = self.ind[pop][i,j] + self.mut_amp*(self.ind[pop][rand_1,j]-self.ind[pop][rand_2,j])
                #Turnering
                for i in range(int(self.no_parents[pop]/2)):
                    randint = np.random.randint(0,self.population_size_list[pop])
                    off_value = self.eval_likelihood_ind(self.offspring[i])
                    if off_value<self.likely_ind[pop][randint]:
                        self.ind[pop][randint] = self.offspring[i]
                        self.likely_ind[pop][randint] = off_value
                
        if method == 'DE':
            for pop in range(self.no_pop):
                self.F_list[pop][:] = 0.1
                self.CR_list[pop][:] = 0.1
                self.u = np.zeros_like(self.ind[pop])
                self.v = np.zeros_like(self.ind[pop])
                klai = np.argsort(self.likely_ind[pop], axis = 0)
                best_index = klai[0]
                ind_best = self.ind[pop][best_index]
                self.abs_best = self.likely_ind[pop][best_index]
                population = self.population_size_list[pop]
                
                for i in range(population):
                    rand1_ = np.random.randint(population)
                    rand2_ = np.random.randint(population)
                    rand3_ = np.random.randint(population)
                    
                    #rand/1 scheme
                    self.v[i] = self.ind[pop][rand1_] + self.F_list[pop][i] * (self.ind[pop][rand2_]- self.ind[pop][rand3_])
                
                for i in range(self.population_size_list[pop]):
                    for j in range(self.dim):
                        randint = np.random.randint(0,1)
                        if randint < self.CR_list[pop][i]:
                            self.u[i,j] = self.v[i,j]
                    if self.eval_likelihood_ind(self.u[i]) < self.likely_ind[pop][i]:
                        self.ind[pop][i] = self.u[i]
        
        if method == 'jDE':
            for pop in range(self.no_pop):
                self.u = np.zeros_like(self.ind[pop])
                self.v = np.zeros_like(self.ind[pop])
                klai = np.argsort(self.likely_ind[pop], axis = 0)
                best_index = klai[0]
                ind_best = self.ind[pop][best_index]
                self.abs_best = self.likely_ind[pop][best_index]
                population = self.population_size_list[pop]
                
                for i in range(population):
                    rand1_ = np.random.randint(population)
                    rand2_ = np.random.randint(population)
                    rand3_ = np.random.randint(population)
                    
                    #rand/1 scheme
                    self.v[i] = self.ind[pop][rand1_] + self.F_list[pop][i] * (self.ind[pop][rand2_]- self.ind[pop][rand3_])
                
                
                for i in range(self.population_size_list[pop]):
                    for j in range(self.dim):
                        randint = np.random.randint(0,1)
                        if randint < self.CR_list[pop][i]:
                            self.u[i,j] = self.v[i,j]
                    if self.eval_likelihood_ind(self.u[i]) < self.likely_ind[pop][i]:
                        self.ind[pop][i] = self.u[i]
        
                for i in range(self.population_size_list[pop]):
                    random_uni_1 = np.random.uniform(0,1)
                    random_uni_2 = np.random.uniform(0,1)
                    random_uni_3 = np.random.uniform(0,1)
                    random_uni_4 = np.random.uniform(0,1)
                    if random_uni_1 < self.tau1:
                        self.F_list[pop][i] += random_uni_2*self.F_list[pop][i]
                    if random_uni_3 < self.tau2:
                        self.CR_list[pop][i] =random_uni_4
        
        if method == 'jDErpo':
            for pop in range(self.no_pop):
                NP = int(self.population_size_list[pop] * np.random.uniform(0,1)+1)
                #Må legge til +1 for å ikke ødelegge random int fra lenger ned
                self.u = np.zeros_like(self.ind[pop])
                self.v = np.zeros_like(self.ind[pop])
                klai = np.argsort(self.likely_ind[pop], axis = 0)
                best_indexes = klai[0:NP]
                xpbest = self.ind[pop][best_indexes]
                #self.abs_best = self.likely_ind[pop][best_indexes[0]]
                population = self.population_size_list[pop]
                
                #Mutant vector
                for i in range(population):
                    rand1_ = np.random.randint(population)
                    rand2_ = np.random.randint(population)
                    rand3_ = np.random.randint(population)
                    rand4_ = np.random.randint(NP)
                    if np.random.uniform(0,1) < 0.8 and self.iter > 0.8*self.maxiter:
                        #rand/1 scheme
                        self.v[i] = self.ind[pop][rand1_] + self.F_list[pop][i]* (self.ind[pop][rand2_]- self.ind[pop][rand3_])
                    else:   
                        #pBest/1 scheme
                        self.v[i] = xpbest[rand4_] + self.F_list[pop][i]*(self.ind[pop][rand2_]- self.ind[pop][rand3_])
                
                Flow  = 0.2  + 0.3  * self.iter/self.maxiter
                CRlow = 0.05 + 0.95 * self.iter/self.maxiter
                
                for i in range(self.population_size_list[pop]):
                    random_uni_1 = np.random.uniform(0,1)
                    random_uni_2 = np.random.uniform(0,1)
                    random_uni_3 = np.random.uniform(0,1)
                    random_uni_4 = np.random.uniform(0,1)
                    if random_uni_1 < self.tau1:
                        self.F_list[pop][i] =Flow + random_uni_2*(self.Fupp - Flow)
                    if random_uni_3 < self.tau2:
                        self.CR_list[pop][i] = CRlow + random_uni_3*(self.CRupp - CRlow)
                
                for i in range(self.population_size_list[pop]):
                    for j in range(self.dim):
                        randint = np.random.randint(0,1)
                        if randint < self.CR_list[pop][i]:
                            self.u[i,j] = self.v[i,j]
                    if self.eval_likelihood_ind(self.u[i]) < self.likely_ind[pop][i]:
                        self.ind[pop][i] = self.u[i]
  
        if method == 'JADE':                
            S_CR = []
            S_F = []
            delta_f = []
            for pop in range(self.no_pop):
                self.F_list[pop][:] = 0.1
                self.CR_list[pop][:] = 0.1
                population_size = self.population_size_list[pop]
                p_i = np.random.uniform(2/population_size, 0.2)
                NP = int(population_size * p_i)        
                H = population_size
                
                #Som i jDE
                self.u = np.zeros_like(self.ind[pop])
                self.v = np.zeros_like(self.ind[pop])
                klai = np.argsort(self.likely_ind[pop], axis = 0)
                best_indexes = klai[0:NP]
                xpbest = self.ind[pop][best_indexes]
                self.abs_best = self.likely_ind[pop][best_indexes[0]]
                
                #Mutant vector
                for i in range(population_size):
                    r_i = np.random.randint(1,H) 
                    self.CR_list[pop][i] = np.random.normal(self.M_CR[pop][r_i], 0.1)
                    #Burde være Cauchy-fordeling
                    self.F_list[pop][i]  = np.random.normal(self.M_F[pop][r_i], 0.1)
                    
                    #Current to pbest/1 
                    rand1_ = np.random.randint(population_size)
                    rand2_ = np.random.randint(population_size)
                    rand3_ = np.random.randint(NP)
                    self.v[i] = self.ind[pop][i] + self.F_list[pop][i]*(xpbest[rand3_]-self.ind[pop][i]) + self.F_list[pop][i]*(self.ind[pop][rand1_]- self.ind[pop][rand2_])
                                                    
                #Crossover
                for i in range(population_size):
                    for j in range(self.dim):
                        randint = np.random.randint(0,1)
                        if randint < self.CR_list[pop][i]:
                            self.u[i,j] = self.v[i,j]
                    if self.eval_likelihood_ind(self.u[i]) <= self.likely_ind[pop][i]:
                        self.ind[pop][i] = self.u[i]
                    # if self.eval_likelihood_ind(self.u[i]) < self.likely_ind[pop][i]:
                        self.A.append(self.ind[pop][i])        
                        delta_f.append(self.eval_likelihood_ind(self.u[i])-self.likely_ind[pop][i])                
                        S_CR.append(self.CR_list[pop][i])
                        S_F.append(self.F_list[pop][i])
                        self.ind[pop][i] = self.u[i]
                        if len(self.A) > self.population_size_list[pop]:
                            del self.A[np.random.randint(0, population_size)]
                #Update weights
                if len(S_CR) != 0:
                    if self.k_arg>=H:
                        self.k_arg = 1
                    wk = []
                    mcr = 0
                    mf_nom = 0
                    mf_denom = 0
                    for arg in range(len(S_CR)):
                        wk.append(delta_f[arg]/sum(delta_f))
                    for arg in range(len(S_CR)):
                        mcr += wk[arg] * S_CR[arg]
                        mf_nom  += wk[arg]*S_F[arg]**2
                        mf_denom += wk[arg]*S_F[arg]
                    self.M_CR[pop][self.k_arg] = mcr
                    self.M_F[pop][self.k_arg] = mf_nom/mf_denom
                    self.k_arg += 1
        
        if method == 'SABLA':
                        
            for pop in range(self.no_pop):   
                if pop == 0:         
                    #Evolve pop  -> cpop
                    #Evaluate pop  
                    #Combine parent (pop) and child (cpop) -> rpop

                    population = self.population_size_list[pop]
                    self.F_list[pop][:] = 0.1
                    self.CR_list[pop][:] = 0.1
                    self.u = np.zeros_like(self.ind[pop])
                    self.v = np.zeros_like(self.ind[pop])
                    klai = np.argsort(self.likely_ind[pop], axis = 0)
                    best_index = klai[0]
                    ind_best = self.ind[pop][best_index]
                    self.abs_best = self.likely_ind[pop][best_index]
                    
                    for i in range(population):
                        rand1_ = np.random.randint(population)
                        rand2_ = np.random.randint(population)
                        rand3_ = np.random.randint(population)
                        
                        #rand/1 scheme
                        self.v[i] = self.ind[pop][rand1_] + self.F_list[pop][i] * (self.ind[pop][rand2_]- self.ind[pop][rand3_])
                    
                    for i in range(self.population_size_list[pop]):
                        for j in range(self.dim):
                            randint = np.random.randint(0,1)
                            if randint < self.CR_list[pop][i]:
                                self.u[i,j] = self.v[i,j]
                        if self.eval_likelihood_ind(self.u[i]) < self.likely_ind[pop][i]:
                            self.ind[pop][i] = self.u[i]
                    self.eval_likelihood_pop(self.ind[pop])
                    self.rpop = np.argsort(self.likely_ind[pop], axis = 0)
                elif pop == 1:                
                    population = self.population_size_list[pop]
                    self.F_list[pop][:] = 0.1
                    self.CR_list[pop][:] = 0.1
                    self.u = np.zeros_like(self.ind[pop])
                    self.v = np.zeros_like(self.ind[pop])
                    klai = np.argsort(self.likely_ind[pop], axis = 0)
                    best_index = klai[0]
                    ind_best = self.ind[pop][best_index]
                    self.abs_best = self.likely_ind[pop][best_index]
                    
                    for i in range(population):
                        rand1_ = np.random.randint(population)
                        rand2_ = np.random.randint(population)
                        rand3_ = np.random.randint(population)
                        
                        #rand/1 scheme
                        self.v[i] = self.ind[pop][rand1_] + self.F_list[pop][i] * (self.ind[pop][rand2_]- self.ind[pop][rand3_])
                    
                    for i in range(self.population_size_list[pop]):
                        for j in range(self.dim):
                            randint = np.random.randint(0,1)
                            if randint < self.CR_list[pop][i]:
                                self.u[i,j] = self.v[i,j]
                        if self.eval_likelihood_ind(self.u[i]) < self.likely_ind[pop][i]:
                            self.ind[pop][i] = self.u[i]
                    
                    
                    #Rank rpop -> rpop
                    #Reduce rpop
                    #Update archive
                    #rank(pop, archive) -> pop
                    #reduce(pop) -> pop
                    #Update best x_u, x_l, F_u, f_l as best in arhive
                #return best x_u, x_l, F_u, f_l
                
        if method == 'SHADE':                
            S_CR = []
            S_F = []
            delta_f = []
            for pop in range(self.no_pop):
                self.F_list[pop][:] = 0.1
                self.CR_list[pop][:] = 0.1
                population_size = self.population_size_list[pop]
                p_i = np.random.uniform(2/population_size, 0.2)
                NP = int(population_size * p_i)        
                H = population_size
                
                #Som i jDE
                self.u = np.zeros_like(self.ind[pop])
                self.v = np.zeros_like(self.ind[pop])
                klai = np.argsort(self.likely_ind[pop], axis = 0)
                best_indexes = klai[0:NP]
                xpbest = self.ind[pop][best_indexes]
                self.abs_best = self.likely_ind[pop][best_indexes[0]]
                
                #Mutant vector
                for i in range(population_size):
                    r_i = np.random.randint(1,H) 
                    self.CR_list[pop][i] = np.random.normal(self.M_CR[pop][r_i], 0.1)
                    #Burde være Cauchy-fordeling
                    self.F_list[pop][i]  = np.random.normal(self.M_F[pop][r_i], 0.1)
                    
                    #Current to pbest/1 
                    rand1_ = np.random.randint(population_size)
                    rand2_ = np.random.randint(population_size)
                    rand3_ = np.random.randint(NP)
                    self.v[i] = self.ind[pop][i] + self.F_list[pop][i]*(xpbest[rand3_]-self.ind[pop][i]) + self.F_list[pop][i]*(self.ind[pop][rand1_]- self.ind[pop][rand2_])
                                                    
                #Crossover
                for i in range(population_size):
                    for j in range(self.dim):
                        randint = np.random.randint(0,1)
                        if randint < self.CR_list[pop][i]:
                            self.u[i,j] = self.v[i,j]
                    if self.eval_likelihood_ind(self.u[i]) <= self.likely_ind[pop][i]:
                        self.ind[pop][i] = self.u[i]
                    # if self.eval_likelihood_ind(self.u[i]) < self.likely_ind[pop][i]:
                        self.A.append(self.ind[pop][i])        
                        delta_f.append(self.eval_likelihood_ind(self.u[i])-self.likely_ind[pop][i])                
                        S_CR.append(self.CR_list[pop][i])
                        S_F.append(self.F_list[pop][i])
                        self.ind[pop][i] = self.u[i]
                        if len(self.A) > self.population_size_list[pop]:
                            del self.A[np.random.randint(0, population_size)]
                #Update weights
                if len(S_CR) != 0:
                    if self.k_arg>=H:
                        self.k_arg = 1
                    wk = []
                    mcr = 0
                    mf_nom = 0
                    mf_denom = 0
                    for arg in range(len(S_CR)):
                        wk.append(delta_f[arg]/sum(delta_f))
                    for arg in range(len(S_CR)):
                        mcr += wk[arg] * S_CR[arg]
                        mf_nom  += wk[arg]*S_F[arg]**2
                        mf_denom += wk[arg]*S_F[arg]
                    self.M_CR[pop][self.k_arg] = mcr
                    self.M_F[pop][self.k_arg] = mf_nom/mf_denom
                    self.k_arg += 1
        
    def select_offspring(self):        
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
    

    def migration(self, migr_prob):
        rand_ = np.random.uniform(0,1)
        if self.no_pop>1:
            if rand_ < migr_prob:
                rand_list = []
                for pop in range(self.no_pop):
                    rand_list.append(np.random.randint(0,len(self.likely_ind[pop])))
                
                for pop in range(self.no_pop,2):                    
                    var = self.ind[pop][rand_list[pop],:]
                    self.ind[pop][rand_list[pop],:] = self.ind[pop+1][rand_list[pop+1],:]
                    self.ind[pop+1][rand_list[pop+1],:] = var
                    
                    
    def nomadic(self):
        pop_saver = []
        if len(pop_saver)>10:
            del pop_saver[0]
        pop_saver.append(self.ind)
        
        
        
        #Ha en oversikt over hvorvidt en populasjon er litt stagnant, men innenfor 2sigma
            #Covarians matrise?
        #Lag en perturbasjons-operator for å få denne populasjonen til å bevege på seg
        
        #Eventuell indre repulsjonskraft
        
        
        return 0                    
    
    
    def repulsion(self):
        if self.conv:
            repulsion = 1e-2
            for pop in range(self.no_pop):
                for i in range(self.population_size_list[pop]):
                    for j in range(self.population_size_list[pop]):
                        if np.abs(self.ind[pop][i]- self.ind[pop][j]) < 1e-2:
                            self.ind[pop][j] += repulsion
                            self.ind[pop][i] -= repulsion
                    