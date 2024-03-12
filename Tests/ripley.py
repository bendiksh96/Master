import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d




def surface_function(x,y):
#    Rosenbrock
    func = 100*(y-x**2)**2 + (1 - x)**2

#    Himmmelblau
    # func = (x**2 + y - 11)**2 + (x + y**2 -7)**2
    # func = np.log(func)
    # func += 1
    
    
    if func < 3.09:
        func = 3.09 + abs(func-3.09)


    return func


def ripley_k_point_to_surface(x, y, num_samples):
    distances = []
    a = np.linspace(0.5,1.5,num_samples)
    for _ in range(num_samples):
        point = np.random.uniform(0.5, 1.5, size=(2,))
        # print(point)
        surface_value = surface_function(*point)
        distance = np.sqrt((x - point[0])**2 + (y - point[1])**2) + surface_value
        distances.append(distance)
    return distances

x_ = np.linspace(0.5,1.5,100)
y_ = np.linspace(0.5,1.5,100)
num_samples = 100

ripley_k_values = []
for y in y_:
    row_distances = []
    for x in x_:
        distances = ripley_k_point_to_surface(x, y, num_samples)
        row_distances.append(np.mean(distances))
    ripley_k_values.append(row_distances)




# Plotting the Ripley-K function
plt.contourf(x_, y_, ripley_k_values, levels=100, cmap='jet') 
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ripley-K Function')
plt.show()