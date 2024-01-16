import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5,5,1000)
y = np.linspace(-5,5,1000)
X,Y = np.meshgrid(x,y)

def Eggholder(x,y):
    func = 0
    func -= (y+47)*np.sin(np.sqrt(abs(y+(x/2)+47)))+ x*np.sin(np.sqrt(abs(x-(y+47))))
    return func


def Himmelblau(x, y):
    func = 0
    func += (x**2 + y - 11)**2 + (x + y**2 -7)**2
    func += 1
    func = np.log(func)
    return func

plt.contourf(x,y, Himmelblau(X,Y))
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour in 2D of Himmelblau function')
plt.colorbar()
plt.savefig(r'C:\Users\Lenovo\Documents\Master\Figs\Himmelblau_contour.pdf')