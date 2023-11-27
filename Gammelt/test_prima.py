import numpy as np
import matplotlib
import matplotlib.pyplot as plt

iter = 100
np_ = 10000
n_p = 500
m_p = 0.1
x_min = -2
x_max = 2
dim = 3

x = np.zeros((np_ , dim))
x_val = np.zeros((np_))
x_off = np.zeros((int(n_p/2), dim))

def rosenbrock(x):
    func = 0
    for i in range(dim-1):        
        func +=  100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2
    return func

def crossover(x,x_val, j):
    offspring =[] 
    sort = np.argsort(x_val)
    x_par = x[sort]
    for k in range(dim):
        randint = np.random.randint(0,1)
        if randint==1:
            offspring.append(x_par[j,k])
        else: 
            offspring.append(x_par[j+1,k])
    return offspring

for i in range(np_):
    for j in range(dim):
        rand_ = np.random.uniform(x_min, x_max)
        x[i,j] = rand_
        
for i in range(iter):
    for j in range(np_):
        x_val[j] = rosenbrock(x[j])
    for j in range(int(n_p/2)):
        x_off[j] = crossover(x,x_val, j)
    for j in range(int(n_p/2)):
        randint = np.random.randint(0, np_)
        x[randint] = x_off[j]

    for j in range(np_):
        mutant = np.zeros(dim)
        for k in range(dim):
            mutant[k] = (np.random.randint(0,np_))
        rand = np.random.uniform(0,1)
        if rand < m_p:
            a = int(mutant[0])
            aa = int(mutant[1])
            aaa = int(mutant[2])
            x[j] = x[a] + 0.1*(x[aa]- x[aaa])
    if not i%10:
        print(i)

klip = np.where(x_val > 6)
x_val[klip] = 'nan'
x[klip] = 'nan'

norm = matplotlib.cm.colors.Normalize(vmin=0, vmax=6)
cmap = plt.get_cmap("viridis_r", 6)

fig, axs = plt.subplots(2,2)
axs[0,0].scatter(x[:,0],x[:,1], c=x_val, norm=norm, cmap=cmap)
axs[0,1].scatter(x[:,1],x[:,2], c=x_val, norm=norm, cmap=cmap)
axs[1,0].scatter(x[:,0],x[:,2], c=x_val, norm=norm, cmap=cmap)
plt.show()
