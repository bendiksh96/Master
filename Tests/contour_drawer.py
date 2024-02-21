import numpy as np
import matplotlib.pyplot as plt

dim = 2


def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

x = np.linspace(-.5,.5,1000)
y = np.linspace(-.5,.5,1000)

X,Y = np.meshgrid(x,y)
Z = rosenbrock(X,Y)
plt.contour(X,Y, Z, levels= 1000, cmap = 'viridis_r')
plt.colorbar()
plt.show()