# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:26:42 2024

@author: Bendik Selvaag-Hagen
"""
import sys
import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
import csv
import time
def conditions(problem_function):
    if problem_function == 'Himmelblau':
        xmin, xmax = -5,5
    if problem_function == 'Eggholder':
        xmin, xmax = -512,512
    if problem_function == 'Rosenbrock':
        xmin, xmax = -5,5
    if problem_function == 'Michalewicz':
        xmin, xmax = 0, np.pi
    if problem_function == 'Rotated_Hyper_Ellipsoid':
        xmin, xmax = -50,50
    if problem_function == 'Hartman_3D':
        xmin, xmax = 0,1
    if problem_function == 'Levy':
        xmin, xmax = -10, 10
    if problem_function == 'Rastrigin':
        xmin, xmax = -5,5 
    if problem_function == 'Ackley':
        xmin, xmax = -32, 32
    return xmin, xmax

def Rosenbrock(x):
    func = 0
    for i in range(dim-1):
        func += 100*(x[i+1]-x[i]**2)**2 + (1 - x[i])**2
    
    return func
    
def Ackley(x):
    a = 20
    b = 0.2
    c = 2*np.pi
    func = a + np.exp(1)
    sum1 = 0
    sum2 = 0
    for i in range(dim):
        sum1 += x[i]**2
        sum2 += np.cos(c*x[i])
    func -= a * np.exp(-b * np.sqrt(sum1 / dim))
    func -= np.exp(sum2 / dim)
    
    func += 1
    func = np.log(func)
    return func

def Rastrigin(x):
    func = 10 * dim
    for i in range(dim):
        func += x[i]**2 - 10 * np.cos(2 * np.pi * x[i])
    return func

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


def Eggholder(x):
    func = 0
    for i in range(dim-1):
        func -= (x[i+1]+47)*np.sin(np.sqrt(abs(x[i+1]+(x[i]/2)+47)))+ x[i]*np.sin(np.sqrt(abs(x[i]-(x[i+1]+47))))
    
    if len(x) == 2:
        func += 959.6407
    if len(x) == 3:
        func += 1597.278
    if len(x) == 4:
        func += 4043.574
    if len(x) == 5:
        func += 15292.97
    if len(x) == 6:
        func += 36524.74
    return func

def Hartman_3D(x):
    a_matrix = [[3, 10, 30],
                [0.1, 10, 35],
                [3, 10, 30],
                [0.1, 10, 35]]
    alfa = [1, 1.2, 3, 3.2]
    p_matrix = [[0.3689, 0.1170, 0.2673],
                [0.4699, 0.4387, 0.7470],
                [0.1091, 0.8732, 0.5547],
                [0.03815, 0.5743, 0.8828]]

    func = 0
    for i in range(4):
        prod = 0
        for j in range(dim):
            prod += a_matrix[i][j] * (x[j] - p_matrix[i][j])**2
        func -= alfa[i] * np.exp(-prod)
    
    if len(x)==3:
        func += 3.86278
    
    return func

def Michalewicz(x):
    m = 10
    func = 0
    for i in range(dim):
        func -= np.sin(x[i]) * (np.sin(((i+1) * x[i]**2) / np.pi))**(2*m)
    if len(x)==3:
        func += 1.8013
    return func

def Rotated_Hyper_Ellipsoid(x):
    func = 0
    for i in range(dim):
        for j in range(i+1):
            func += x[j]**2
            
    func += 1
    func = np.log(func)
    return func



def Himmelblau(x):
    func = 0
    for i in range(dim-1):
        func += (x[i]**2 + x[i+1] - 11)**2 + (x[i] + x[i+1]**2 -7)**2
    func += 1
    func = np.log(func)

    if len(x) == 2:
        pass
    elif len(x) == 3:
        func -= 0.265331837897597
    elif len(x) == 4:
        func -= 1.7010318616354436
    elif len(x) == 5:
        func -= 2.3001107745553155
    elif len(x) == 6:
        func -= 2.8576426513378994
    else:
        raise Exception("We don't know the minimum value for Himmelblau in this number of dimensions.")

    return func
# path = (r'C:\Users\Lenovo\Documents\Master\Tests\new_data.csv')

def bin_it(problem_function, xmin, xmax, n_bin_per_dim):   
    bin_space = np.linspace(xmin,xmax, n_bin_per_dim+1)
    bin_val = np.zeros((n_bin_per_dim, n_bin_per_dim, n_bin_per_dim))
    
    for u in range(n_bin_per_dim):
        for v in range(n_bin_per_dim):
            val_list = []
            for w in range(n_bin_per_dim):
                mid_0 = bin_space[u] + abs(bin_space[u] - bin_space[u+1])/2
                mid_1 = bin_space[v] + abs(bin_space[v] - bin_space[v+1])/2
                mid_2 = bin_space[w] + abs(bin_space[w] - bin_space[w+1])/2
                mid = np.array((mid_0, mid_1, mid_2))
                
                b_0_min =bin_space[u]
                b_0_max =bin_space[u+1]
                b_1_min =bin_space[v]
                b_1_max =bin_space[v+1]
                b_2_min =bin_space[w]
                b_2_max =bin_space[w+1]

                #Finn GFBS for funksjonen i dette binnet
                if problem_function == 'Rosenbrock':
                    res = sp.minimize(Rosenbrock, mid, method = 'L-BFGS-B', bounds = [(b_0_min,b_0_max),(b_1_min, b_1_max),( b_2_min, b_2_max) ])      
                if problem_function == 'Himmelblau':
                    res = sp.minimize(Himmelblau, mid, method = 'L-BFGS-B', bounds = [(b_0_min,b_0_max),(b_1_min, b_1_max),( b_2_min, b_2_max) ])      
                if problem_function == 'Eggholder':
                    res = sp.minimize(Eggholder, mid, method = 'L-BFGS-B', bounds = [(b_0_min,b_0_max),(b_1_min, b_1_max),( b_2_min, b_2_max) ])      
                if problem_function == 'Michalewicz':
                    res = sp.minimize(Michalewicz, mid, method = 'L-BFGS-B', bounds = [(b_0_min,b_0_max),(b_1_min, b_1_max),( b_2_min, b_2_max) ])      
                if problem_function == 'Rotated_Hyper_Ellipsoid':
                    res = sp.minimize(Rotated_Hyper_Ellipsoid, mid, method = 'L-BFGS-B', bounds = [(b_0_min,b_0_max),(b_1_min, b_1_max),( b_2_min, b_2_max) ])      
                if problem_function == 'Hartman_3D':
                    res = sp.minimize(Hartman_3D, mid, method = 'L-BFGS-B', bounds = [(b_0_min,b_0_max),(b_1_min, b_1_max),( b_2_min, b_2_max) ])      
                if problem_function == 'Levy':
                    res = sp.minimize(Levy, mid, method = 'L-BFGS-B', bounds = [(b_0_min,b_0_max),(b_1_min, b_1_max),( b_2_min, b_2_max) ])      
                if problem_function == 'Rastrigin':
                    res = sp.minimize(Rastrigin, mid, method = 'L-BFGS-B', bounds = [(b_0_min,b_0_max),(b_1_min, b_1_max),( b_2_min, b_2_max) ])      
                if problem_function == 'Ackley':
                    res = sp.minimize(Ackley, mid, method = 'L-BFGS-B', bounds = [(b_0_min,b_0_max),(b_1_min, b_1_max),( b_2_min, b_2_max) ])      
                bin_val[u,v,w] = res.fun
                val_list.append(res.fun)
                 
        print('progress:',u/n_bin_per_dim*100, '%')
    return bin_val
dim = 0
if dim == 4:    
    bin_space = np.linspace(xmin,xmax, n_bin_per_dim+1)
    #print(bin_space)
    bin_val = np.zeros((n_bin_per_dim, n_bin_per_dim, n_bin_per_dim, n_bin_per_dim))
    
    for u in range(n_bin_per_dim):
        for v in range(n_bin_per_dim):
            for w in range(n_bin_per_dim):
                #val_list = []
                for q in range(n_bin_per_dim):
                    #print(u,v,w)
                    mid_0 = bin_space[u] + abs(bin_space[u] - bin_space[u+1])/2
                    mid_1 = bin_space[v] + abs(bin_space[v] - bin_space[v+1])/2
                    mid_2 = bin_space[w] + abs(bin_space[w] - bin_space[w+1])/2
                    mid_3 = bin_space[q] + abs(bin_space[q] - bin_space[q+1])/2
                    mid = np.array((mid_0, mid_1, mid_2, mid_3))
                    #print('midpoint:',mid)
                    
                    b_0_min =bin_space[u]
                    b_0_max =bin_space[u+1]
                    b_1_min =bin_space[v]
                    b_1_max =bin_space[v+1]
                    b_2_min =bin_space[w]
                    b_2_max =bin_space[w+1]
                    b_3_min =bin_space[q]
                    b_3_max =bin_space[q+1]
                    
                    res = sp.minimize(Himmelblau, mid, method = 'L-BFGS-B', bounds = [(b_0_min,b_0_max),(b_1_min, b_1_max),( b_2_min, b_2_max) , (b_3_min, b_3_max)])      
                    #print('result',res.x)
                    #print(res.fun)
                    #print()
                    #print(Himmelblau(res.x))
                    
                    bin_val[u,v,w,q] = res.fun
                   # val_list.append(res.fun)
                #self.path = (r"C:\Users\Lenovo\Documents\Master\datafile.csv")
                #with open(path, 'a', newline='') as csvfile:
                #    csvfile = csv.writer(csvfile, delimiter=',')
                #    csvfile.writerow(val_list)
        print('progress:',u/n_bin_per_dim*100, '%')
#print(bin_val)


path = (r'C:\Users\Lenovo\Documents\Master\Tests\3d_validation_rosenbrock.npy')
dim = 3
n_bin_per_dim = 100

start = time.time()
func_list   =['Ackley']# [ 'Rotated_Hyper_Ellipsoid','Hartman_3D', 'Levy', 'Rastrigin', 'Ackley', 'Rosenbrock', 'Himmelblau' ]#['Eggholder' , 'Michalewicz']
for f in range(len(func_list)):
    path = 'C:/Users/Lenovo/Documents/Master/Tests/3d_' + func_list[f]  + '.npy' 
    xmin, xmax = conditions(func_list[f])
    print("Now performing the collection on ", func_list[f])
    array = bin_it(func_list[f], xmin, xmax, n_bin_per_dim)
    np.save(path, array)
end = time.time()

    
print('Seconds:',end-start)
  

