import numpy as np

def Rosenbrock(x_):
        func = 0
        for i in range(dim-1):
            func += 100*(x_[i+1]-x_[i]**2)**2 + (1 - x_[i])**2
        return func
    
def sigma_(x_, mu):
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
cluster_record = []
def pool(ind, k, gen):
    if k == 1:
        #Create a cluster!
        s_cluster.append([1])
        mu_cluster.append([ind])
        var_cluster.append([0])
        cluster_record.append([k])
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

            mig = mu_12
        
        if k>= 3:
            inhabit = False
            inhabit_ind = []

            for i in range(len(s_cluster)):
                #Update temporary variables
                #9
                s_temp = (s_cluster[i][k-2] +1)
                
                #10
                mu_temp = (s_temp - 1)/s_temp * mu_cluster[i][k-2] + 1/s_temp * ind
                
                #11
                var_temp =  (s_temp - 1)/s_temp * var_cluster[i][k-2] + 1/(s_temp- 1) * (np.linalg.norm(ind-mu_temp))**2

                #7
                elip = 1/s_temp + (ind-mu_temp)@(ind-mu_temp).T / (s_temp* var_temp)
                #8
                reu  =  elip/2
                
                #Add individual to cloud
                m = 3
                if reu <= (m**2+1)/(2*s_temp):
                    s_cluster[i].append(s_temp)
                    mu_cluster[i].append(mu_temp)
                    var_cluster[i].append(var_temp)
                    inhabit = True
                    inhabit_ind.append(i)
                    cluster_record[i].append(k)
                    
                #Update old cloud 
                else:
                    s_cluster[i].append(s_cluster[i][k-2])
                    mu_cluster[i].append(mu_cluster[i][k-2])
                    var_cluster[i].append(var_cluster[i][k-2])
                    cluster_record[i].append(cluster_record[i][k-2])            
        
            #Om individ ikke tilhører noen skyer, lag en ny
            if inhabit != True:
                #Er det k-2 her?
                arg_length = [0 for a in range (k-2)]
                var_cluster.append(arg_length)
                cluster_record.append(arg_length)
                cluster_record[-1].append(k)
                
                arg_length[-1] = 1
                s_cluster.append(arg_length)
                arg_length[-1] = ind
                mu_cluster.append(arg_length)
                inhabit_ind.append(i+1)
                
                cluster_record.append(arg_length)
                cluster_record[-1].append(k)
                
            
            for i in range(len(s_cluster)):
                for j in range(len(s_cluster)):
                    if i != j:
                        print(i,j)

                        count = 0
                        rez = {}
                        print(cluster_record[j])
                        exit()
                        for elem in cluster_record[i][-1]:
                            rez[count] = cluster_record[j][-1].count(elem)
                            count +=1
                        print(rez)
                        print()
                        a = np.where(cluster_record[i][-1] == cluster_record[j][-1])

                        intersect = len(cluster_record[i][-1][a])
                        if 2*intersect > s_cluster[i][-1] or 2*intersect > s_cluster[j][-1]:
                            print('Tjohei')
                            
                        #Intersection(s_i and s_j) * 2> s_i or s_j:
                            #start merging
                            #15
                            #16
                            #17           
        
        #Mean individual fra en cloud der individet er 
            ran = np.random.randint(0, len(inhabit_ind))
            mig = mu_cluster[ran][k-2]    
    
    #Returner k og individet
    k += 1 
    return mig, k

num_pop     = 2
pop_size    = 100
NFE_max     = 1e7
dim         = 5
NFE         = 0 
T           = 1e-3
sigma       = []
mu          = []
individual  = []
likelihood  = []
lmd         = []
xmin,xmax   = -5,5
gen         = 0 
role        = 'island'
roly        = []
k           = 1

mu_initial = np.zeros((num_pop, dim))
sig_initial = np.zeros((num_pop, dim))
for pop in range(num_pop):
    role_pop = []
    kar = np.zeros((pop_size,dim))
    verdi = np.zeros((pop_size,1))
    for ind in range(pop_size):
        role_pop.append('island')
        for j in range(dim):
            kar[ind,j] = np.random.uniform(xmin, xmax)
            mu_initial[pop,j] = np.mean(kar[:,j])
            sig_initial[pop,j] = sigma_(kar[:,j], mu_initial[pop, j])
        verdi[ind,0] = Rosenbrock(kar[ind,:])        
    roly.append(role_pop)
    individual.append(kar)
    likelihood.append(verdi)
    mu.append(mu_initial)
    sigma.append(sig_initial)
    


while NFE < NFE_max:
    NFE += pop_size*num_pop
    gen += 1
    if role == 'island':
        mu_arr  = np.zeros((num_pop,dim))
        sig_arr = np.zeros((num_pop,dim))
        tht_arr = np.zeros((num_pop,dim))
        omg_arr = np.zeros((num_pop,dim))
        lmd_arr = np.zeros((num_pop,dim))
        tauh_arr= np.zeros((num_pop,dim))
        taub_arr= np.zeros((num_pop,dim))
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
                if Rosenbrock(u[i]) < likelihood[pop][i]:
                    individual[pop][i] = u[i]
                    
            for j in range(dim):
                #Her må du finne ut hva som trenger array og ikke
                mu_arr[pop, j] = np.mean(individual[pop][:, j])
                sig_arr[pop,j] = sigma_(individual[pop][:,j], mu_arr[pop,j])
                mu_min = np.mean(mu[gen-1])
                tht_arr[pop,j] = theta_val(mu_arr[pop,j], mu_min, sig_arr[pop,j])
                omg_arr[pop,j] = omega(tht_arr[pop,j])  
                tauh_arr[pop,j] = tau_hat(sig_arr[pop,j],omg_arr[pop,j]) 
                lmd_arr[pop,j] = lmd_val(lmd_arr[pop,j],mu_arr[pop,j],mu[gen-1][pop,j], sig_arr[pop,j], sigma[gen-1][pop,j])
                taub_arr[pop,j] = tau_bar(lmd_arr[pop,j]) 
                tau_arr[pop,j] = tau_tot(tauh_arr[pop,j], taub_arr[pop,j])
        mu.append(mu_arr)    
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
