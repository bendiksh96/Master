import numpy as np
import matplotlib.pyplot as plt


def conditions(problem_function):
    if problem_function == 'Himmelblau':
        xmin, xmax = -5,5
    if problem_function == 'Rosenbrock':
        xmin, xmax = -5,10
    if problem_function == 'Levy':
        xmin, xmax = -10, 10
    if problem_function == 'Rastrigin':
        xmin, xmax = -5,5 
    return xmin, xmax
dim = 2

def Levy(x):
    w = []
    for i in range(dim):
        w.append(1 + (x[i]-1)/4)
    term1 = (np.sin(np.pi * w[0]))**2
    term_sum = 0
    for i in range(dim-1):
        term_sum += ((w[i] - 1)**2) * (1 + 10 * (np.sin(np.pi * w[i] + 1))**2)
    
    term_end = ((w[dim-1] - 1)**2) * (1 + (np.sin(2 * np.pi * w[dim-1])**2))
    
    func = term1 + term_sum + term_end
    return func
    
met_list = ['Himmelblau', 'Rosenbrock', 'Levy', 'Rastrigin']

xmin, xmax = conditions(met_list[2])
x = np.linspace(xmin, xmax, 100)
y = np.linspace(xmin, xmax, 100)

X,Y = np.meshgrid(x,y)

plt.contourf(x,y, c = Levy(x,y))