import numpy as np
import sys 
sys.path.append(r"C:\Users\Lenovo\Documents\GitHub\Master\Physics")
from physical_eval import *

class dSHADEpso:
    def __init__(self, num_ind, ggd_list):
        self.num_ind        = num_ind
        self.dim            = 3

        self.A          = []
        self.Flist      = [0.1 for p in range(self.num_ind)]
        self.CRlist     = [0.1 for p in range(self.num_ind)]
        self.M_CR       = [0.1 for p in range(self.num_ind)]
        self.M_F        = [0.1 for p in range(self.num_ind)]

        self.k_arg      = 0 
        self.ggd_list   = ggd_list
        self.xs             = XSection(ggd_list)
        self.nfe            = 0
        self.hist_data      = []
        self.modifier       = 0
    
    def initialize_population(self):
        #gluino, neutralino, squark
        self.ind_ind    = ['g', 'n' , 'q']
        self.individual = np.zeros((self.num_ind, self.dim))
        self.likelihood = np.zeros(self.num_ind)
        self.optimal_individual = np.zeros_like(self.individual)
        self.velocity = np.zeros_like(self.individual)
        self.force = np.zeros_like(self.individual)
        self.v          = np.zeros_like(self.individual)
        self.xmin_arr = [500 ,  0 , 500]
        self.xmax_arr = [3000,2000,3000]
        
        #Initialize 
        for p in range(self.num_ind):
            for j in range(self.dim):
                if self.ind_ind[j] == 'g' or self.ind_ind[j] == 'q':
                    self.individual[p, j] = np.random.uniform(500,3000)
                if self.ind_ind[j] == 'n':
                    self.individual[p, j] = np.random.uniform(0  ,2000)
            temp, true_likelihood, signal, section = self.eval_likelihood_ind(self.individual[p])
            k = [self.individual[p,j] for j in range(self.dim)] + [true_likelihood] + [signal] + [section]
            self.hist_data.append(k)    
            self.likelihood[p] = temp
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
                    temp, true_likelihood, signal, section = self.eval_likelihood_ind(self.v[i])
                    k = [self.v[i,j] for j in range(self.dim)] + [true_likelihood] + [signal] + [section]
                    self.hist_data.append(k)
                    self.nfe += 1
                    if temp < self.likelihood[i]:
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
            if randint < self.CRlist[i] or randu < 0.3:                
                self.v[i], candidate_status = self.check_oob(self.v[i])
                if candidate_status:
                    temp, true_likelihood, signal, section = self.eval_likelihood_ind(self.v[i])
                    k = [self.v[i,j] for j in range(self.dim)] + [true_likelihood] + [signal] + [section]
                    self.hist_data.append(k)
                    self.nfe += 1
                    if temp < self.likelihood[i]:
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
    


    def cluster(self, loglike_tol, k):
        """_summary_

        Args:
            k (_int_): number of clusters 
            loglike_tol (_float_): threshold of the log-likelihood
            
        Returns:
            super_centroids (_array_): The center of mass of each centroid 
            super_labels (_array_): The label of each individual, corresponding to a centroid
        """
        self.nfe = 0
        self.k = k
        self.un_empty_cluster = k
        cluster_sort = np.where(self.likelihood< (loglike_tol+.5))
        self.X = self.individual[cluster_sort]

        labels    = np.zeros(self.k)
        self.super_centroids = np.zeros((k,self.dim))
        self.super_labels = np.zeros((self.num_ind))
        maxiter = 1000
        centroids = self.X[np.random.choice(range(len(self.X)), size=self.k, replace=False)]
        
        for _ in range(maxiter):
            labels = np.zeros((self.num_ind))
            distance = np.zeros((self.num_ind, self.k))
            
            #Finn avstand til alle punkter
            for w in range(k):
                for i in range(self.num_ind):
                    distance[i,w] = np.linalg.norm(self.individual[i,:] - centroids[w])
                
            #Har avstand til alle punkter
            
            #Skriv disse 
            for i in range(self.num_ind):
                max_dist = 1000
                for w in range(self.k):
                    if distance[i,w] < max_dist:
                        labels[i] = int(w)
                        max_dist = distance[i,w]
            new_centroids = np.zeros_like(centroids)
            for w in range(self.k):                
                new_dex = np.where(labels[:] == w)
                new_ind = self.individual[new_dex]
                new_centroids[w] = np.mean(new_ind[:])
            if np.linalg.norm(new_centroids == centroids) < 1e-3:
                self.super_centroids = centroids
                self.super_labels[:] = labels[:]
                break
            centroids = new_centroids
        self.super_centroids = centroids
        self.super_labels = labels
        
        
        #Evaluate Centroids
        for arg in range(self.k):
            cl, status = self.check_oob(self.super_centroids[arg,:])
            if status == True:
                perceived_likelihood,true_likelihood,signal, section  = self.eval_likelihood_ind(cl)
                k = [self.v[i,j] for j in range(self.dim)] + [true_likelihood] + [signal] + [section]
                self.hist_data.append(k)
                self.nfe += 1

            krev = np.where(self.super_labels == arg)
            krev_ = krev[0]
            
            for ab in range(len(krev_)):
                i = krev_[ab]
                if true_likelihood > self.likelihood[i]:
                    true_likelihood = self.likelihood[i]
                    self.super_centroids[arg] = self.individual[i]
                    
            lik_tresh = 5.915
            if true_likelihood > lik_tresh:
                mess = np.where(self.super_labels == arg)
                temp_lik = self.likelihood[mess]
                new_lik = 'nan'

                for b in range(len(temp_lik)):
                    if temp_lik[b] < true_likelihood:
                        new_lik = temp_lik[b]
                        true_likelihood = new_lik
                        self.super_centroids[arg, :] = new_lik
                        
                #No individuals in cluster with sufficiently low likelihood
                if new_lik == 'nan':
                    temp_ind = self.individual[mess]
                    for b in range(len(temp_lik)):
                        dist_cluster = 100
                        for argy in range(k):
                            if arg != argy: 
                                #Oppdater tilhørighet gitt nærmeste cluster
                                dist_comp = np.linalg.norm(temp_ind[b]- self.super_centroids[argy])
                                if dist_comp < dist_cluster:
                                    self.super_labels[mess] = argy
                                    dist_cluster = dist_comp        

                    #Regn ut ny cluster (med disse nye labels inkludert)
                    #Lag ny cluster array
                    self.un_empty_cluster -= 1
        for i in range(self.k-1):
            ##?
            perceived_likelihood, true_likelihood, _, _  = self.eval_likelihood_ind(self.super_centroids[i])
            #Noen av centroids suger.            
        return self.super_centroids, self.super_labels
    
    def init_particles(self):
        #Velocity initialization
        for i in range(self.num_ind):            
            for j in range(self.dim):
                #ru = np.random.uniform(int(-1e-3),int(1e-3))
                ru = np.random.uniform(0,1) 
                ru = ru*0.0001
                self.velocity[i,j] = (self.individual[i,j] - self.optimal_individual[i,j]) * ru 
        print('Particles Initialized')
    
    def evolve_particle(self):
        self.nfe = 0
        for i in range(self.num_ind):
            k = .01
            for j in range(self.dim):
                self.force[i,j]         = -k * (self.likelihood[i] - self.optimum) * (self.individual[i,j] - self.optimal_individual[i,j])#/abs(self.individual[i,j] - self.optimal_individual[i,j])
                self.velocity[i,j]      = self.velocity[i,j] + self.force[i,j]
                self.individual[i,j]    = self.individual[i,j] + self.velocity[i,j]
            
            self.individual[i,:], candidate_status = self.check_oob(self.individual[i])
            
            if candidate_status:
                self.likelihood[i], true_likelihood = self.eval_likelihood_ind(self.individual[i])
                k = [self.individual[i,j] for j in range(self.dim)] + [true_likelihood] + [3]
                self.hist_data.append(k)
                # print(self.likelihood[i], true_likelihood)
                self.true_likelihood[i] = true_likelihood
                self.nfe += 1        
      

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
        percieved_val, true_val, signal, section = self.xs.evaluate(individual)
        return percieved_val, true_val, signal, section
    
    