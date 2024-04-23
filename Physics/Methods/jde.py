import numpy as np

class jDE:
    def __init__(self, individual, likelihood):
        self.problem_func   = problem_func
        self.individual     = individual
        self.likelihood     = likelihood
        self.num_ind        = len(likelihood)
        self.dim            = len(individual[0])
        self.u              = np.zeros_like(self.individual)
        self.v              = np.zeros_like(self.individual)
        p_i                 = np.random.uniform(2/self.num_ind, 0.2)
        NP                  = int(self.num_ind * p_i)        
        H                   = self.num_ind
        self.Flist          = [0.1 for p in range(self.num_ind)]
        self.CRlist         = [0.1 for p in range(self.num_ind)]
        self.tau1,self.tau2 = 0.1,0.1
        self.Data           = Problem_Function(self.dim)
        self.nfe            = 0
        self.hist_data      = []

    def evolve(self):
        self.nfe        = 0
        sort_index      = np.argsort(self.likelihood, axis = 0)
        best_index      = sort_index[0]
        best_individual = self.individual[best_index]
        self.abs_best   = self.likelihood[best_index]
        
        for i in range(self.num_ind):
            ri1 = np.random.randint(self.num_ind)
            ri2 = np.random.randint(self.num_ind)
            ri3 = np.random.randint(self.num_ind)
            
            #rand/1 scheme
            self.v[i] = self.individual[ri1] + self.Flist[i] * (self.individual[ri2] - self.individual[ri3])
        
        #Muter        
        for i in range(self.num_ind):
            randint = np.random.uniform(0,1)
            if randint < self.CRlist[i]:
                temp, true_likelihood = self.eval_likelihood_ind(self.v[i])
                k = [self.v[i,j] for j in range(self.dim)] + [true_likelihood] + [1]
                self.hist_data.append(k)
                self.nfe += 1
                if temp < self.likelihood[i]:
                    self.individual[i] = self.v[i]
                    self.likelihood[i] = temp

        #Crossover
        for i in range(self.num_ind):
            ru1 = np.random.uniform(0,1)
            ru2 = np.random.uniform(0,1)
            ru3 = np.random.uniform(0,1)
            ru4 = np.random.uniform(0,1)
            if ru1 < self.tau1:
                self.Flist[i] += ru2 * self.Flist[i]
            if ru3 < self.tau2:
                self.CRlist[i] = ru4
    #Metode for å evaluere likelihood til et enkelt individ.
    #Liker ikke helt å måtte kalle på den her også, men det er hittil det beste jeg har.
    def eval_likelihood_ind(self, individual):
        Data = self.Data
        return Data.evaluate(individual, self.problem_func)
