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



num_pop     = 4
pop_size    = 10
NFE_max     = 1e4
dim         = 3
NFE         = 0 
T           = 1e-3
sigma       = []
mu          = []
individual  = []
likelihood  = []
lmd         = []
xmin,xmax   = -5,5
role        = 'island'
gen         = 0 

mu_initial = np.zeros((num_pop, dim))
sig_initial = np.zeros((num_pop, dim))
for pop in range(num_pop-1):
    kar = np.zeros((pop_size,dim))
    verdi = np.zeros((pop_size,1))
    for ind in range(pop_size):
        for j in range(dim):
            kar[ind,j] = np.random.uniform(xmin, xmax)
            mu_initial[pop,j] = np.mean(kar[:,j])
            sig_initial[pop,j] = sigma_(kar[:,j], mu_initial[pop, j])
        verdi[ind,0] = Rosenbrock(kar[ind,:])        
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
        mig, gamma_1, gamma_2, gamma_3 = 0 
        if NDIV==dim:
            gamma_1 = 1
        
        if any(tau_arr[pop,:] == 1) and np.random.uniform(0,1) < 1e-3:
            gamma_2 = 1
        
        if (NDIV/dim) >= (1- NFE/NFE_max):
            gamma_3 = 1
        
        if gamma_1 or gamma_2 or gamma_3:
            mig = 1
        
        if mig:
            #Migrasjonskriterie møtt
            a = 0
            print('Fett')
    
