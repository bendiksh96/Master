import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
#Himmelblau
x = np.linspace(-5,5,1000)
y = np.linspace(-5,5,1000)
X,Y = np.meshgrid(x,y)

def Himmelblau(x, y):
    func = 0
    func += (x**2 + y - 11)**2 + (x + y**2 -7)**2
    func += 1
    func = np.log(func)
    return func

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def levy(x,y):
    w = []
    w.append(1 + (x-1)/4)
    w.append(1 + (y-1)/4)
    term1 = (np.sin(np.pi * w[0]))**2
    term_sum = 0
    term_sum += ((w[0] - 1)**2) * (1 + 10 * (np.sin(np.pi * w[0] + 1))**2)
    term_sum += ((w[1] - 1)**2) * (1 + 10 * (np.sin(np.pi * w[1] + 1))**2)
    
    term_end = ((w[0] - 1)**2) * (1 + (np.sin(2 * np.pi * w[0])**2))
    
    func = term1 + term_sum + term_end
    return func

def rastrigin(x,y):
    func = 20
    func += x**2 - 10 * np.cos(2 * np.pi * x)
    func += y**2 - 10 * np.cos(2 * np.pi * y)
    return func


# plt.contourf(x,y, rosenbrock(X,Y), levels =10, cmap = 'viridis_r')
# plt.grid()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Contour in 2D of Himmelblau function')
# plt.colorbar()
# # plt.savefig(r'C:\Users\Lenovo\Documents\Master\Figs\Himmelblau_contour.pdf')
# plt.show()


# Plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Himmelblau(X,Y), cmap='viridis_r')
ax.set_title('Himmelblau Function')
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_zlabel('f$(x_0,x_1)$')
cax = fig.add_axes([0.15, 0.2, 0.03, 0.5])
cbar = fig.colorbar(surf, cax=cax, label = 'f($x_0,x_1$)', aspect = 5)
cbar.ax.yaxis.set_label_coords(-3, 0.5)
cbar.ax.yaxis.set_ticks_position('left')
plt.savefig(r'C:\Users\Lenovo\Documents\GitHub\Master\Figs\himmelblau _shape.png')

#Rosenbrock
x = np.linspace(-5,10,1000)
y = np.linspace(-5,10,1000)
X,Y = np.meshgrid(x,y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y,np.log(rosenbrock(X,Y)), cmap='viridis_r')
ax.set_title('Rosenbrock Function')
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_zlabel('f$(x_0,x_1)$')
cax = fig.add_axes([0.15, 0.2, 0.03, 0.5])
cbar = fig.colorbar(surf, cax=cax, label = '$\ln$f($x_0,x_1$)', aspect = 5)
cbar.ax.yaxis.set_label_coords(-3, 0.5)
cbar.ax.yaxis.set_ticks_position('left')
plt.savefig(r'C:\Users\Lenovo\Documents\GitHub\Master\Figs\rosenbrock_shape.png')


#Levy
x = np.linspace(-32,32,1000)
y = np.linspace(-32,32,1000)
X,Y = np.meshgrid(x,y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, levy(X,Y), cmap='viridis_r')
ax.set_title('Levy Function')
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_zlabel('f$(x_0,x_1)$')
cax = fig.add_axes([0.15, 0.2, 0.03, 0.5])
cbar = fig.colorbar(surf, cax=cax, label = 'f($x_0,x_1$)', aspect = 5)
cbar.ax.yaxis.set_label_coords(-3, 0.5)
cbar.ax.yaxis.set_ticks_position('left')
plt.savefig(r'C:\Users\Lenovo\Documents\GitHub\Master\Figs\levy_shape.png')


#Rastrigin
x = np.linspace(-5,5,1000)
y = np.linspace(-5,5,1000)
X,Y = np.meshgrid(x,y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, rastrigin(X,Y), cmap='viridis_r')
ax.set_title('Rastrigin Function')
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_zlabel('f$(x_0,x_1)$')
cax = fig.add_axes([0.15, 0.2, 0.03, 0.5])
cbar = fig.colorbar(surf, cax=cax, label = 'f($x_0,x_1$)', aspect = 5)
cbar.ax.yaxis.set_label_coords(-3, 0.5)
cbar.ax.yaxis.set_ticks_position('left')
plt.savefig(r'C:\Users\Lenovo\Documents\GitHub\Master\Figs\rastrigin_shape.png')
