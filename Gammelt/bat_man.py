import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
    
num_pop     = 4
pop_size    = 1000
NFE_max     = 1e4
dim         = 5
NFE         = 0 
BM          = [] 
BM_val      = []
individual  = []
likelihood  = []
velocity    = []
pulse       = []
xmin,xmax   = -512,512
gen         = 0 
rj          = .2
rj0         = .2
A           = [.5 for  p in range(num_pop)]
eps         = .01
gamma       = .9
alfa        = .9

#A - loudness
#Needs to be tuned in order to 


for pop in range(num_pop):
    role_pop = []
    kar = np.zeros((pop_size,dim))
    vel = np.zeros((pop_size,dim))
    puls = np.zeros((pop_size,dim))
    verdi =np.ones(pop_size)
    for i in range(pop_size):
        for j in range(dim):
            kar[i,j] = np.random.uniform(xmin, xmax)
            vel[i,j] = xmax/dim*np.random.uniform(-1,1)
            puls[i,j] = rj
        verdi[i] = Eggholder(kar[i,:])
    individual.append(kar)
    likelihood.append(verdi)
    velocity.append(vel)
    pulse.append(puls)
    
    sort = np.argsort(verdi)
    BM.append(kar[sort])
    BM_val.append(verdi[sort])
    
gen = 1
while NFE < NFE_max:
    for pop in range(num_pop):
        sort = np.argsort(likelihood[pop])
        best_ind = individual[pop][sort][0]
        best_val = likelihood[pop][sort][0]
        worst_val = likelihood[pop][sort][-1]
        #Forslag 7D
        # super_val = -5216
        # best_profile = super_val + (pop+1)/2*(super_val - best_val)
        for i in range(pop_size):
            for j in range(dim):
                u = np.random.uniform(0,1)
                #Frequency
                f_j = best_val + (best_val-worst_val) * np.random.uniform(0,1)
                #Velocity
                # velocity[pop][i,j] = velocity[pop][i,j] + 1/(100*pop+pop_size)*(individual[pop][i,j] - best_ind[0][j]) * f_j[0][0]

                velocity[pop][i,j] = velocity[pop][i,j] +(individual[pop][i,j] - best_ind[j]) * f_j
                #Update individual if
                if u < pulse[pop][i,j]:
                    individual[pop][i,j] = best_ind[j] + eps*A[pop]
                else:
                    individual[pop][i,j] = velocity[pop][i,j] + individual[pop][i,j] 
                
                #oob
                if individual[pop][i,j] < xmin:
                    if velocity[pop][i,j] < 0:
                        velocity[pop][i,j] = - velocity[pop][i,j]
                    if individual[pop][i,j] < 1.1*xmin:
                        individual[pop][i,j] = np.random.uniform(xmin,xmax)
                
                if individual[pop][i,j] > xmax:
                    if velocity[pop][i,j] > 0:
                        velocity[pop][i,j] = - velocity[pop][i,j]
                    if individual[pop][i,j] > 1.1*xmax:
                        individual[pop][i,j] = np.random.uniform(xmin,xmax)
                    

            likelihood[pop][i] = Eggholder(individual[pop][i])

            #If the likelihood is equal, increase velocity
            if any(abs(likelihood[pop][i] - BM_val[pop][i]) < [1e-3 for p in range(dim)]):
                velocity[pop][i] = velocity[pop][i] +A[pop]
            
            #If likelihood smaller or loudness over threshold -> Update BM and reduce loudness
            if likelihood[pop][i] < BM_val[pop][i] or np.random.uniform(0,1) < A[pop]:
                pulse[pop][i] = pulse[pop][i]*(1- np.exp(-gamma*gen))
                A[pop] = alfa*A[pop]
                BM[pop][i] = individual[pop][i]
                BM_val[pop][i] = likelihood[pop][i]           
        #Sort the bats
        sort = np.argsort(BM_val[pop])
        BM[pop] = BM[pop][sort]
        BM_val[pop] = BM_val[pop][sort]
        
        
    gen += 1
    NFE += num_pop * pop_size


# plt.scatter( BM[0][:, 0],  BM[0][:,1])#, 'b', label = 'Population 1')
# plt.scatter( BM[1][:, 0],  BM[1][:,1],  BM_val[1][:], 'r', label = 'Population 2')
# plt.scatter( BM[2][:, 0],  BM[2][:,1],  BM_val[2][:], 'g', label = 'Population 3')
# plt.scatter( BM[3][:, 0],  BM[3][:,1],  BM_val[3][:], 'c', label = 'Population 4')

# #plt.legend( )
# plt.xlabel('x0')
# plt.ylabel('x1')
# plt.title('')
# plt.ylim(xmin, xmax)
# plt.xlim(xmin,xmax)
# plt.title('Bat Algorithm on Eggholder. Unnomralized.')  
# plt.show()

for i in range(pop_size):
    print(BM[0][i], BM_val[0][i])
# print(BM)
# print(BM_val)

from vis_writ import * 
klam = Data(dim, xmin, xmax)
klam.visualize_1( BM[0],  BM_val[0], eval_type = 'Bat')
