import numpy as np
import sys 
sys.path.append(r"C:\Users\Lenovo\Documents\GitHub\Master\Physics")
from physical_eval   import *
class dSHADE:
    def __init__(self, num_ind, ggd_list):
        self.num_ind        = num_ind
        self.dim            = 3
        self.A          = []
        self.Flist      = [0.1 for p in range(self.num_ind)]
        self.CRlist     = [0.1 for p in range(self.num_ind)]
        self.M_CR       = [0.1 for p in range(self.num_ind)]
        self.M_F        = [0.1 for p in range(self.num_ind)]

        self.k_arg      = 0 

        self.ggd_list       = ggd_list
        self.xs             = XSection(ggd_list)
        
        self.nfe            = 0
        self.hist_data      = []
        self.modifier       = 0
    
    def initialize_population(self):
        #gluino, neutralino, squark
        self.ind_ind    = ['g', 'n' , 'q']
        self.individual = np.zeros((self.num_ind, self.dim))
        self.likelihood = np.zeros(self.num_ind)
        self.v          = np.zeros_like(self.individual)
        self.xmin_arr   = [500 ,  0 , 500]
        self.xmax_arr   = [3000,2000,3000]
        
        #Initialize 
        for p in range(self.num_ind):
            for j in range(self.dim):
                if self.ind_ind[j] == 'g' or self.ind_ind[j] == 'q':
                    self.individual[p, j] = np.random.uniform(500,3000)
                if self.ind_ind[j] == 'n':
                    self.individual[p, j] = np.random.uniform(0  ,2000)
            pred,true, pv_1, tv_1, s1,pv_2, tv_2, s2, pv_3, tv_3, s3, pv_4, tv_4, s4, section = self.eval_likelihood_ind(self.individual[p])
            k = [self.individual[p,j] for j in range(self.dim)] + [pred] + [true] +[pv_1] + [tv_1] + [s1]  + [pv_2] + [tv_2] + [s2] + [pv_3] + [tv_3] + [s3] + [pv_4] + [tv_4] + [s4] +[section]
            self.hist_data.append(k)
            self.nfe += 1
            self.likelihood[p] = pred
        self.nfe = self.num_ind

    def evolve_converge(self):
        S_CR       = []
        S_F        = []
        delta_f    = []

        self.v          = np.zeros_like(self.individual)
        sort            = np.argsort(self.likelihood, axis = 0)
        best_indexes    = sort[0:self.num_ind]
        self.abs_best   = self.likelihood[best_indexes[0]]
        self.best_ind   = self.individual[best_indexes][0]
        xpbest          = self.individual[best_indexes]
        for i in range(self.num_ind):
            ri = np.random.randint(1,self.num_ind) 
            self.CRlist[i] = np.random.normal(self.M_CR[ri], 0.1)
            #Burde være Cauchy-fordeling
            self.Flist[i]  = np.random.normal(self.M_F[ri], 0.1)
            
            #Current to pbest/1 
            ri1 = np.random.randint(self.num_ind)
            ri2 = np.random.randint(self.num_ind)
            ri3 = np.random.randint(self.num_ind/4)
            
            self.v[i] = self.individual[i] + self.Flist[i]*(xpbest[ri3]-self.individual[i]) + self.Flist[i]*(self.individual[ri1]- self.individual[ri2])

        #Muter        
        for i in range(self.num_ind):
            randint = np.random.uniform(0,1)
            randu   = np.random.uniform(0,1)
            if randint < self.CRlist [i] or randu < 0.3:                
                self.v[i], candidate_status = self.check_oob(self.v[i])
                if candidate_status:
                    pred, true, pv_1, tv_1, s1,pv_2, tv_2, s2, pv_3, tv_3, s3, pv_4, tv_4, s4, section = self.eval_likelihood_ind(self.v[i])

                    k = [self.v[i,j] for j in range(self.dim)] +[pred] + [true]+ [pv_1] + [tv_1] + [s1]  + [pv_2] + [tv_2] + [s2] + [pv_3] + [tv_3]+ [s3] + [pv_4] + [tv_4] + [s4] +[section]
                    self.hist_data.append(k)
                    self.nfe += 1
                    temp = pv_1 + pv_2 + pv_3 + pv_4
                    if temp < self.likelihood[i]or randu < 0.3:
                        self.individual[i] = self.v[i]
                        self.likelihood[i] = temp

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

    def evolve_explore(self):
        S_CR       = []
        S_F        = []
        delta_f    = []

        self.v          = np.zeros_like(self.individual)
        sort            = np.argsort(self.likelihood, axis = 0)
        best_indexes    = sort[0:self.num_ind]
        xpbest          = self.individual[best_indexes]
        for i in range(self.num_ind):
            ri = np.random.randint(1,self.num_ind) 
            self.CRlist[i] = np.random.normal(self.M_CR[ri], 0.1)
            #Burde være Cauchy-fordeling
            self.Flist[i]  = np.random.normal(self.M_F[ri], 0.1)
            
            #pbest-to-pbest/1:
            ri1 = np.random.randint(self.num_ind)
            rii = np.random.randint(self.num_ind/2)
            ri3 = np.random.randint(self.num_ind/2)
            self.v[i] = xpbest[rii] + self.Flist[i]*(xpbest[ri3]-self.individual[ri1])

        #Muter        
        for i in range(self.num_ind):
            randint = np.random.uniform(0,1)
            randu   = np.random.uniform(0,1)
            if randint < self.CRlist[i]:                
                self.v[i], candidate_status = self.check_oob(self.v[i])
                if candidate_status:
                    pred,true,pv_1, tv_1, s1,pv_2, tv_2, s2, pv_3, tv_3, s3, pv_4, tv_4, s4, section = self.eval_likelihood_ind(self.v[i])
                    k = [self.v[i,j] for j in range(self.dim)] +[pred] + [true]+ [pv_1] + [tv_1] + [s1]  + [pv_2] + [tv_2] + [s2] + [pv_3] + [tv_3]+ [s3] + [pv_4] + [tv_4] + [s4] +[section]
                    self.hist_data.append(k)
                    self.nfe += 1
                    temp = pv_1 + pv_2 + pv_3 + pv_4
                    if temp < self.likelihood[i]or randu < 0.3:
                        self.individual[i] = self.v[i]
                        self.likelihood[i] = temp

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
            xmin, xmax = self.xmin_arr[j], self.xmax_arr[j]
            if candidate[j] < xmin:
                var = xmin - (candidate[j] - xmin)
                if var > xmax or var < xmin:
                    candidate_status = False
                else:
                    candidate[j] = var
                    
            if  candidate[j] > xmax:
                var  = xmax - (candidate[j] - xmax)
                if var < xmin or var > xmax:
                    candidate_status = False
                else:
                    candidate[j] = var
        return candidate, candidate_status
    
    def alter_likelihood(self):
        self.xs.modifier = self.modifier
        self.xs.abs_best = self.abs_best
        self.xs.conv_    = True
    
    def eval_likelihood_ind(self, individual):
        if len(self.ggd_list) == 1:
            percieved_val, true_val, signal, section = self.xs.evaluate(individual)
            return percieved_val, true_val, signal, section
        if len(self.ggd_list) == 4:
            pred, true,pv_1, tv_1, s1,pv_2, tv_2, s2, pv_3, tv_3, s3, pv_4, tv_4, s4, section = self.xs.evaluate(individual)
            return pred,true, pv_1, tv_1, s1,pv_2, tv_2, s2, pv_3, tv_3, s3, pv_4, tv_4, s4, section
