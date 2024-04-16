import numpy as np
import matplotlib.pyplot as plt

dim = 2
def Eggholder(x,dim):
    func = 0
    for i in range(dim-1):
        func -= (x[i+1]+47)*np.sin(np.sqrt(abs(x[i+1]+(x[i]/2)+47)))+ x[i]*np.sin(np.sqrt(abs(x[i]-(x[i+1]+47))))
    return func

x = np.ones(dim)
print(x)

plt.plot(x, Eggholder(x, dim))

    