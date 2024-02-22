import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


def Eggholder(x):
    func = 0
    dim = len(x[0])
    for i in range(dim-1):
        func -= (x[i+1]+47)*np.sin(np.sqrt(abs(x[i+1]+(x[i]/2)+47)))+ x[i]*np.sin(np.sqrt(abs(x[i]-(x[i+1]+47))))
        
    return func


num_ind = 10
dim     = 3
individual = np.zeros((num_ind,dim))
likelihood = np.zeros(num_ind)

for i in range(num_ind):
    for j in range(dim):
        individual[i,j] = np.random.uniform(-5,5)
    likelihood[i] = Eggholder(individual[i,:])

res = sp.optimize.minimize((Eggholder, [0,0,0]))