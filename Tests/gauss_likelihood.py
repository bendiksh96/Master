import numpy as np
import matplotlib.pyplot as plt
import time


def Himmelblau(x):
    func = 0
    for i in range(1):
        func += (x[i]**2 + x[i+1] - 11)**2 + (x[i] + x[i+1]**2 -7)**2
    func += 1
    func = np.log(func)
    best = np.min(func)
    return func, best

def mod_himmelblau(x, best, delta_log):
    func = 0

    #Calculate Himmelblau in 2D
    for i in range(1):
        func += (x[i]**2 + x[i+1] - 11)**2 + (x[i] + x[i+1]**2 -7)**2

    # sigma = np.std([func,best])

    #Make it logarithmic
    func += 1
    func = np.log(func)

    #Sigma is the std. For now random variable
    sigma = delta_log/3

    true_func = func

    #Calculate Gauss on best point.

    if true_func <= (best+delta_log):
        func = 0

        #Calculate Himmelblau in 2D
        for i in range(1):
            func += (x[i]**2 + x[i+1] - 11)**2 + (x[i] + x[i+1]**2 -7)**2

        func = best + delta_log
        func = func  + delta_log * np.exp(- (best-(true_func))**2/(2*sigma**2)) 


    return func #, true_func




x = np.linspace(-5,5, 1000)
y = np.zeros_like(x)

ind = np.zeros((2,len(x)))
for i in range(len(x)):
    ind[0,i] = x[i]
    ind[1,i] = y[i]


dev, best  = Himmelblau(ind)

func_val = np.zeros_like(x)
for i in range(len(x)):
    func_val[i] = mod_himmelblau(ind[:,i], best, 1)


plt.plot(x, dev, 'r-', label = 'Likelihood Himmelblau')
plt.plot(x, func_val, 'b-', label = 'Likelihood Himmelblau w/ Gauss')
plt.legend()
plt.xlabel('x')
plt.ylabel('likelihood')
plt.title('Profile likelihood 2D, y = 0')
plt.show()













