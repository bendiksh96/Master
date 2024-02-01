import numpy as np
import sys

sys.path.append(r'C:\Users\Lenovo\Documents\Master')
from problem_func import * 

class DDMS:
    def __init__(self, individual, likelihood, problem_func):
        self.individual = individual
        self.likelihood = likelihood
        self.u              = np.zeros_like(self.individual)
        self.v              = np.zeros_like(self.individual)

        self.dim            = len(individual[0])        
        self.num_ind        = len(likelihood)
        self.problem_func   = problem_func
        self.Data           = Problem_Function(self.dim)

        self.Flist          = [0.1 for p in range(self.num_ind)]
        self.CRlist         = [0.1 for p in range(self.num_ind)]
        self.tau1,self.tau2 = 0.1,0.1

        
        self.s_cluster   = []
        self.mu_cluster  = []
        self.var_cluster = []
        self.ind_cluster = []
        self.cluster_record = []
        self.inhabit_num = []

        self.mu_prev  = np.zeros((self.dim))    
        self.lmd_prev = np.zeros((self.dim))
        self.sigma_prev = np.zeros((self.dim))

   
        self.k      = 1
        self.T      = 1e-3        
        self.nfe    = 0
        self.nfe_max= 0 
        self.gen    = 0
    def pool(self, ind):
        if self.k == 1:
            #Create a cluster!
            self.s_cluster.append([1])
            self.mu_cluster.append([ind])
            self.var_cluster.append([0])
            self.cluster_record.append([self.k])
            self.ind_cluster.append([ind])
            mig = ind
        else:
            if self.k ==2:
                #Add individual to initial cluster
                s_12    = 2
                mu_12   = (s_12 - 1)/s_12 * self.mu_cluster[0][0]  + (1/s_12)*ind
                var_12  = (s_12 - 1)/s_12 * self.var_cluster[0][0] + (1/(s_12-1))*np.linalg.norm(ind- mu_12)
                
                self.s_cluster[0].append(s_12)
                self.mu_cluster[0].append(mu_12)
                self.var_cluster[0].append(var_12)
                self.cluster_record[0].append(self.k)
                self.ind_cluster[0].append([ind] + self.ind_cluster[0][0])
                mig = mu_12
                self.inhabit_num.append([1])

            
            if self.k>= 3:
                inhabit = False
                #Sjekk om den tilhører noen av skyene, oppdater skyer deretter
                for i in range(len(self.s_cluster)):
                    #Update temporary variables
                    num_in_cloud = self.inhabit_num[i][-1]
                    
                    #9
                    s_temp = (self.s_cluster[i][num_in_cloud] +1)
                    # print(s_temp)
                    #10
                    mu_temp = (s_temp - 1)/s_temp * self.mu_cluster[i][num_in_cloud] + 1/s_temp * ind
                    
                    #11
                    var_temp =  (s_temp - 1)/s_temp * self.var_cluster[i][num_in_cloud] + 1/(s_temp- 1) * (np.linalg.norm(ind-mu_temp))**2

                    #7
                    elip = 1/s_temp + (ind-mu_temp)@(ind-mu_temp).T / (s_temp* var_temp)
                    #8
                    reu  =  elip/2
                    
                    #Add individual to cloud
                    #Dette ser ganske shady ut som variabel
                    m = 1
                    
                    if reu <= (m**2+1)/(2*s_temp):
                        inhabit = True
                        
                        self.s_cluster[i].append(s_temp)
                        self.mu_cluster[i].append(mu_temp)
                        self.var_cluster[i].append(var_temp)
                        self.inhabit_num[i][-1] += 1
                        
                        self.cluster_record[i].append(self.k)
                        self.ind_cluster[i].append([ind] + self.ind_cluster[i][num_in_cloud])    
                        ##############SHADY
                        mig = mu_temp
                    #Update old cloud, without adding individual 
                    else:
                        self.s_cluster[i].append(self.s_cluster[i][num_in_cloud])
                        self.mu_cluster[i].append(self.mu_cluster[i][num_in_cloud])
                        self.var_cluster[i].append(self.var_cluster[i][num_in_cloud])
                        self.ind_cluster[i].append(self.ind_cluster[i][num_in_cloud])
                        self.cluster_record[i].append(self.cluster_record[i][num_in_cloud])         
                
                    
                #Om individ ikke tilhører noen skyer, lag en ny
                if inhabit != True:
                    self.s_cluster.append([1])
                    self.mu_cluster.append([ind])
                    self.var_cluster.append([0])
                    self.ind_cluster.append([ind])
                    self.cluster_record.append([self.k])
                    self.inhabit_num.append([0])
                    ###############SHADY
                    mig =  ind 
                
                #Restart Mechanism:
                #Om det er fler enn én sky
                if len(self.s_cluster) > 1:
                    count = 0
                    #For hver cloud
                    for i in range(len(self.s_cluster)):
                        if self.k in self.cluster_record[i]:
                            count += 1
                    #Individet er i alle skyene 
                    if count == len(self.s_cluster):
                        #Perturber hvert individ i skyen
                        for i in range(len(self.s_cluster)):
                            for j in range(len(self.s_cluster[i])):
                                self.ind_cluster[i][j] = (np.random.normal(self.mu_cluster[i][-1],np.sqrt(self.var_cluster[i][-1])) 
                                                    + np.random.uniform(0,1)*ind
                                                    - np.random.uniform(0,1)*self.ind_cluster[i][j])
                    
        self.k += 1 
        
        #Returner individet
        #Men hvilket
        return mig
 
 
    def evolve(self):
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
            for j in range(self.dim):
                randint = np.random.randint(0,1)
                if randint < self.CRlist[i]:
                    self.u[i,j] = self.v[i,j]
            temp, z = self.eval_likelihood_ind(self.u[i])
            if temp < self.likelihood[i]:
                self.individual[i] = self.u[i]
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
        self.test_migration_criteria()
        
    def test_migration_criteria(self):
        mu_arr  = np.zeros((self.dim))
        sig_arr = np.zeros((self.dim))
        lmd_arr = np.zeros((self.dim))
        tau_arr = np.zeros((self.dim))

        mu_prev  = self.mu_prev    
        lmd_prev = self.lmd_prev
        sigma_prev = self.sigma_prev

        gen = self.gen
        
        for j in range(self.dim-1):
            mu_     = np.mean(self.individual[:, j])
            sig_    = self.sigma_func(self.individual[:,j], mu_)
            mu_min  = np.mean(mu_prev)
            tht_    = self.theta_val(mu_, mu_min, sig_)
            omg_    = self.omega(tht_)  
            tauh_   = self.tau_hat(sig_,omg_)                 
            lmd_    = self.lmd_val(lmd_prev[j], mu_ ,mu_prev[j], sig_, sigma_prev[j])
            taub_   = self.tau_bar(lmd_) 
            tau_    = self.tau_tot(tauh_, taub_)

            lmd_arr[j]  = lmd_
            mu_arr[j]   = mu_
            sig_arr[j]  = sig_
            tau_arr[j]  = tau_

        self.mu_prev  = mu_arr    
        self.lmd_prev = lmd_arr
        self.sigma_prev = sig_arr
        
        NDIV = sum(tau_arr[:])
        mig, gamma_1, gamma_2, gamma_3 = 0,0,0,0
        
        if NDIV==self.dim:
            gamma_1 = 1
        
        if any(tau_arr[:] == 1) and np.random.uniform(0,1) < 2e-1:
            gamma_2 = 1
        
        if (NDIV/self.dim) >= (1- self.nfe/self.nfe_max):
            gamma_3 = 1
        
        if gamma_1 or gamma_2 or gamma_3:
            mig = 1
        
        #Migrasjonskriterie møtt
        if mig:
            sort = np.argsort(self.likelihood)
            best_ind = self.individual[sort][0]
            mig_ind = self.pool(best_ind)
            
            new_ind = np.zeros_like(tau_arr)
            for j in range(self.dim-1):
                if tau_arr[j]:
                    new_ind[j] = mig_ind[0][j]
                else:
                    new_ind[j] = best_ind[j]

            self.individual[int(np.random.randint(0,self.num_ind-1))] = new_ind    
    


    def sigma_func(self,x_, mu):
        ret = 0 
        for i in range(self.num_ind):
            ret += np.sqrt((1/self.num_ind)*(x_[i]-mu)**2)
        return ret

    def omega(self,theta):
        if theta > self.T:
            ret = self.T
        else:
            ret = theta
        return ret 

    def theta_val(self,mu, mu_min,sigma):
        if sigma <= self.T:
            ret = abs(mu - mu_min)* self.T
        else:
            ret = self.T
        return ret

    def tau_hat(self, sigma, theta):
        if sigma <= theta:
            return 1
        else:
            return 0

    def lmd_val(self,lmd, mu, mu_prev, sig, sig_prev):
        if mu == mu_prev and sig == sig_prev:
            return lmd+1
        else:
            return 0
        
    def tau_bar(self,lmd):
        if lmd >= self.num_ind:
            return 1
        else:
            return 0

    def tau_tot(self, th, tb):
        if th==1 or tb ==1:
            return 1
        else:
            return 0

    def eval_likelihood_ind(self, individual):
        Data = self.Data
        return Data.evaluate(individual, self.problem_func)






#Merging
"""
#Merging mechanism
merge_list = []
new_list = []
merge_bool = False
for i in range(len(self.s_cluster)):
    for j in range(len(self.s_cluster)):
        if i != j:
            merge_index = []
            for ii in range(len(self.cluster_record[i])):
                for jj in range(len(self.cluster_record[j])):
                    if self.cluster_record[i][ii] == self.cluster_record[j][jj]:
                        merge_index.append(ii)

            intersect = len(merge_index)
            if 2*intersect > self.s_cluster[i][-1] or 2*intersect > self.s_cluster[j][-1]:
                #15
                s_new   = self.s_cluster[i][-1] + self.s_cluster[j][-1] - intersect   
                if s_new > 0:
                    merge_list.append([i, j])
                    merge_bool = True
                    #16
                    mu_new  = (self.s_cluster[i][-1] * self.mu_cluster[i][-1] + self.s_cluster[j][-1] * self.mu_cluster[j][-1])/(self.s_cluster[i][-1]+self.s_cluster[j][-1])
                    #17           
                    var_new = ((self.s_cluster[i][-1] - 1)*self.var_cluster[i][-1] +  (self.s_cluster[j][-1] - 1)*self.var_cluster[j][-1])/(self.s_cluster[i][-1] + self.s_cluster[j][-1] - 2 ) 
                    new_list.append([s_new, mu_new,var_new])                      
                break
merge_num = len(merge_list)
if merge_bool:
    mergers = []
    #Add new, merged clouds
    for u in range(merge_num):
        i,j = merge_list[u][:]
        if i not in mergers:
            mergers.append(i)
        if j not in mergers:
            mergers.append(j)
                        
        self.s_cluster.append([new_list[u][0]])
        self.mu_cluster.append([new_list[u][1]])
        self.var_cluster.append([new_list[u][2]])
        
        self.ind_cluster.append([ind])
        self.cluster_record.append([self.k])
        self.inhabit_num.append([0])
    
    #Delete old clouds
    count = 0 
    # print('Liste:',mergers)
    for w in mergers:
        w -= count
        del(self.s_cluster[w]); del(self.mu_cluster[w]); del(self.var_cluster[w])
        del(self.ind_cluster[w]); del(self.cluster_record[w]); del(self.inhabit_num[w])
        count += 1
#Mean individual fra en cloud der individet er 
ran = np.random.randint(0, len(self.mu_cluster))
mig = self.mu_cluster[ran][self.inhabit_num[ran][-1]]    
"""
