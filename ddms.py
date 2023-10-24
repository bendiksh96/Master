import numpy as np

np.random.seed(12313)
def Eggholder(x_):
    func = 0
    for i in range(dim-1):
        func -= (x_[i+1]+47)*np.sin(np.sqrt(abs(x_[i+1]+(x_[i]/2)+47)))+ x_[i]*np.sin(np.sqrt(abs(x_[i]-(x_[i+1]+47))))
    
    return func

def Rosenbrock(x_):
        func = 0
        for i in range(dim-1):
            func += 100*(x_[i+1]-x_[i]**2)**2 + (1 - x_[i])**2
        return func
    
def sigma_func(x_, mu):
    ret = 0 
    for i in range(pop_size):
        ret += np.sqrt((1/pop_size)*(x_[j]-mu)**2)
    return ret

def omega(theta):
    if theta > T:
        ret = T
    else:
        ret = theta
    return ret 

def theta_val(mu, mu_min,sigma):
    if sigma <= T:
        ret = abs(mu - mu_min)* T
    else:
        ret = T
    return ret

def tau_hat(sigma, theta):
    if sigma <= theta:
        return 1
    else:
        return 0

def lmd_val(lmd, mu, mu_prev, sig, sig_prev):
    if mu == mu_prev and sig == sig_prev:
        return lmd+1
    else:
        return 0
    
def tau_bar(lmd):
    if lmd >= pop_size:
        return 1
    else:
        return 0

def tau_tot(th, tb):
    if th==1 or tb ==1:
        return 1
    else:
        return 0
#A New Evolving Clustering Algorithm for Online Data Streams, Bezerra et. al
k           = 1
s_cluster   = []
mu_cluster  = []
var_cluster = []
ind_cluster = []
cluster_record = []
inhabit_num = []


def pool(ind, k, gen):
    if k == 1:
        #Create a cluster!
        s_cluster.append([1])
        mu_cluster.append([ind])
        var_cluster.append([0])
        cluster_record.append([k])
        ind_cluster.append([ind])
        mig = ind
    else:
        if k ==2:
            #Add individual to initial cluster
            s_12    = 2
            mu_12   = (s_12 - 1)/s_12 * mu_cluster[0][0]  + (1/s_12)*ind
            var_12  = (s_12 - 1)/s_12 * var_cluster[0][0] + (1/(s_12-1))*np.linalg.norm(ind- mu_12)
            
            s_cluster[0].append(s_12)
            mu_cluster[0].append(mu_12)
            var_cluster[0].append(var_12)
            cluster_record[0].append(k)
            ind_cluster[0].append([ind] + ind_cluster[0][0])
            mig = mu_12
            inhabit_num.append([1])

        
        if k>= 3:
            inhabit = False
            inhabit_ind = []
            for i in range(len(s_cluster)):
                #Update temporary variables
                num_in_cloud = inhabit_num[i][-1]
                
                #9
                s_temp = (s_cluster[i][num_in_cloud] +1)
                print(s_temp)
                #10
                mu_temp = (s_temp - 1)/s_temp * mu_cluster[i][num_in_cloud] + 1/s_temp * ind
                
                #11
                var_temp =  (s_temp - 1)/s_temp * var_cluster[i][num_in_cloud] + 1/(s_temp- 1) * (np.linalg.norm(ind-mu_temp))**2

                #7
                elip = 1/s_temp + (ind-mu_temp)@(ind-mu_temp).T / (s_temp* var_temp)
                #8
                reu  =  elip/2
                
                #Add individual to cloud
                m = 1
                if reu <= (m**2+1)/(2*s_temp):
                    inhabit = True
                    
                    s_cluster[i].append(s_temp)
                    mu_cluster[i].append(mu_temp)
                    var_cluster[i].append(var_temp)
                    inhabit_ind.append(i)
                    inhabit_num[i][-1] += 1
                    
                    cluster_record[i].append(k)
                    ind_cluster[i].append([ind] + ind_cluster[i][num_in_cloud])    
                
                #Update old cloud, without adding individual 
                else:
                    s_cluster[i].append(s_cluster[i][num_in_cloud])
                    mu_cluster[i].append(mu_cluster[i][num_in_cloud])
                    var_cluster[i].append(var_cluster[i][num_in_cloud])
                    ind_cluster[i].append(ind_cluster[i][num_in_cloud])
                    cluster_record[i].append(cluster_record[i][num_in_cloud])         
            
                
            #Om individ ikke tilhører noen skyer, lag en ny
            if inhabit != True:
                s_cluster.append([1])
                mu_cluster.append([ind])
                var_cluster.append([0])
                ind_cluster.append([ind])
                cluster_record.append([k])
                inhabit_num.append([0])
                inhabit_ind.append(i+1)
            
            #Restart Mechanism 
            if len(s_cluster) > 1:
                for i in range(len(s_cluster)):
                    if len(s_cluster[i]) > 10:
                        mask = np.mean( (s_cluster[i][-10]  - s_cluster[i][-1])-s_cluster[i][-10])
                        if mask == 5:
                            #Initiate restart 
                            for j in range(len(s_cluster[i])):
                                #Perturbe the individuals in the cluster                        
                                ind_cluster[i][j] = (np.random.normal(mu_cluster[i][-1],np.sqrt(var_cluster[i][-1])) 
                                                    + np.random.uniform(0,1)*ind
                                                    - np.random.uniform(0,1)*ind_cluster[i][j])
                
            #Merging mechanism
            merge_list = []
            new_list = []
            merge_bool = False
            for i in range(len(s_cluster)):
                for j in range(len(s_cluster)):
                    if i != j:
                        merge_index = []
                        for ii in range(len(cluster_record[i])):
                            for jj in range(len(cluster_record[j])):
                                if cluster_record[i][ii] == cluster_record[j][jj]:
                                    merge_index.append(ii)

                        intersect = len(merge_index)
                        if 2*intersect > s_cluster[i][-1] or 2*intersect > s_cluster[j][-1]:

                            #15
                            s_new   = s_cluster[i][-1] + s_cluster[j][-1] - intersect   
                            print('new:',s_new)   
                            if s_new > 0:
                                merge_list.append([i, j])
                                merge_bool = True
                                #16
                                mu_new  = (s_cluster[i][-1] * mu_cluster[i][-1] + s_cluster[j][-1] * mu_cluster[j][-1])/(s_cluster[i][-1]+s_cluster[j][-1])
                                #17           
                                var_new = ((s_cluster[i][-1] - 1)*var_cluster[i][-1] +  (s_cluster[j][-1] - 1)*var_cluster[j][-1])/(s_cluster[i][-1] + s_cluster[j][-1] - 2 ) 
                                new_list.append([s_new, mu_new,var_new])                      

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
                                    
                    s_cluster.append([new_list[u][0]])
                    mu_cluster.append([new_list[u][1]])
                    var_cluster.append([new_list[u][2]])
                    
                    ind_cluster.append([ind])
                    cluster_record.append([k])
                    inhabit_num.append([0])
                    inhabit_ind.append([len(s_cluster)-merge_num])
                
                #Delete old clouds
                count = 0 
                for w in mergers:
                    w -= count
                    del(s_cluster[w]); del(mu_cluster[w]); del(var_cluster[w])
                    del(ind_cluster[w]); del(cluster_record[w]); del(inhabit_num[w]); del(inhabit_ind[w])

        #Mean individual fra en cloud der individet er 
            ran = np.random.randint(0, len(inhabit_ind))
            mig = mu_cluster[ran][inhabit_num[ran][-1]]    
    
    #Returner k og individet
    k += 1 
    return mig, k

num_pop     = 4
pop_size    = 100
NFE_max     = 1e6
dim         = 5
NFE         = 0 
T           = 1e-3
sigma       = []
mu          = []
lmd         = []
individual  = []
likelihood  = []
xmin,xmax   = -5,5
gen         = 0 
role        = 'island'
roly        = []
k           = 1

mu_initial = np.zeros((num_pop, dim))
sig_initial = np.zeros((num_pop, dim))
lmd_initial = np.zeros((num_pop, dim))
for pop in range(num_pop):
    role_pop = []
    kar = np.zeros((pop_size,dim))
    verdi = np.zeros((pop_size,1))
    for ind in range(pop_size):
        role_pop.append('island')
        for j in range(dim):
            kar[ind,j]          = np.random.uniform(xmin, xmax)
            mu_initial[pop,j]   = np.mean(kar[:,j])
            sig_initial[pop,j]  = sigma_func(kar[:,j], mu_initial[pop, j])
        verdi[ind,0] = Eggholder(kar[ind,:])        
    roly.append(role_pop)
    individual.append(kar)
    likelihood.append(verdi)
    mu.append(mu_initial)
    sigma.append(sig_initial)
    lmd.append(lmd_initial)
    


while NFE < NFE_max:
    NFE += pop_size*num_pop
    gen += 1
    if role == 'island':
        mu_arr  = np.zeros((num_pop,dim))
        sig_arr = np.zeros((num_pop,dim))
        lmd_arr = np.zeros((num_pop,dim))
        tau_arr = np.zeros((num_pop,dim))
        
        for pop in range(num_pop-1):
            #Vanlig propagasjon-scheme
            F  = 0.1
            CR = 0.1
            u = np.zeros_like(individual[pop])
            v = np.zeros_like(individual[pop])
            klai = np.argsort(likelihood[pop], axis = 0)
            best_index = klai[0]
            ind_best = individual[pop][best_index]
            abs_best = likelihood[pop][best_index]
            
            for i in range(pop_size):
                rand1_ = np.random.randint(pop_size)
                rand2_ = np.random.randint(pop_size)
                rand3_ = np.random.randint(pop_size)
                
                #rand/1 scheme
                v[i] = individual[pop][rand1_] + F * (individual[pop][rand2_] - individual[pop][rand3_])
            
            for i in range(pop_size):
                for j in range(dim):
                    randint = np.random.randint(0,1)
                    if randint < CR:
                        u[i,j] = v[i,j]
                argi = Eggholder(u[i])
                if argi < likelihood[pop][i]:
                    individual[pop][i] = u[i]
                    likelihood[pop][i] = argi
                    
            for j in range(dim-1):
                mu_     = np.mean(individual[pop][:, j])
                sig_    = sigma_func(individual[pop][:,j], mu_)
                mu_min  = np.mean(mu[gen-1])
                tht_    = theta_val(mu_, mu_min, sig_)
                omg_    = omega(tht_)  
                tauh_   = tau_hat(sig_,omg_)                 
                lmd_    = lmd_val(lmd[gen-1][pop,j], mu_ ,mu[gen-1][pop,j], sig_, sigma[gen-1][pop,j])
                taub_   = tau_bar(lmd_) 
                tau_    = tau_tot(tauh_, taub_)

                lmd_arr[pop,j]  = lmd_
                mu_arr[pop,j]   = mu_
                sig_arr[pop,j]  = sig_
                tau_arr[pop,j]  = tau_

        mu.append(mu_arr)    
        lmd.append(lmd_arr)
        sigma.append(sig_arr)

        NDIV = sum(tau_arr[pop,:])
        mig, gamma_1, gamma_2, gamma_3 = 0,0,0,0
        if NDIV==dim:
            gamma_1 = 1
        
        if any(tau_arr[pop,:] == 1) and np.random.uniform(0,1) < 1e-1:
            gamma_2 = 1
        
        if (NDIV/dim) >= (1- NFE/NFE_max):
            gamma_3 = 1
        
        if gamma_1 or gamma_2 or gamma_3:
            mig = 1
        
        #Migrasjonskriterie møtt
        if mig:
            sort = np.argsort(likelihood[pop])
            best_ind = individual[pop][sort][0]
            mig_ind, k = pool(best_ind, k, gen)
            individual[pop][int(np.random.randint(0,pop_size-1))] = mig_ind

print(individual)