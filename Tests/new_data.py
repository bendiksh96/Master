# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:26:42 2024

@author: Bendik Selvaag-Hagen
"""

import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
import csv
import time
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
path = (r'C:\Users\Bendik Selvaag-Hagen\OneDrive - Universitetet i Oslo\Documents\GitHub\Master\new_data.csv')

#with open(path, 'w', newline='') as csvfile:
#    pass

dim = 3
xmin, xmax = -5,5
n_bin_per_dim = 100
start = time.time()

if dim == 3:    
    bin_space = np.linspace(xmin,xmax, n_bin_per_dim+1)
    #print(bin_space)
    bin_val = np.zeros((n_bin_per_dim, n_bin_per_dim, n_bin_per_dim))
    
    for u in range(n_bin_per_dim):
        for v in range(n_bin_per_dim):
            val_list = []
            for w in range(n_bin_per_dim):
                #print(u,v,w)
                mid_0 = bin_space[u] + abs(bin_space[u] - bin_space[u+1])/2
                mid_1 = bin_space[v] + abs(bin_space[v] - bin_space[v+1])/2
                mid_2 = bin_space[w] + abs(bin_space[w] - bin_space[w+1])/2
                mid = np.array((mid_0, mid_1, mid_2))
                #print('midpoint:',mid)
                
                b_0_min =bin_space[u]
                b_0_max =bin_space[u+1]
                b_1_min =bin_space[v]
                b_1_max =bin_space[v+1]
                b_2_min =bin_space[w]
                b_2_max =bin_space[w+1]
                #print('bounds',[b_0_min,b_0_max], [b_1_min, b_1_max], [b_2_min, b_2_max])
                #print(sp.minimize(Himmelblau, mid))
                #Finn GFBS for funksjonen i dette binnet
                res = sp.minimize(Himmelblau, mid, method = 'L-BFGS-B', bounds = [(b_0_min,b_0_max),(b_1_min, b_1_max),( b_2_min, b_2_max) ])      
                #print('result',res.x)
                #print(res.fun)
                #print()
                #print(Himmelblau(res.x))
                
                bin_val[u,v,w] = res.fun
                val_list.append(res.fun)
            #self.path = (r"C:\Users\Lenovo\Documents\Master\datafile.csv")
            with open(path, 'a', newline='') as csvfile:
                csvfile = csv.writer(csvfile, delimiter=',')
                csvfile.writerow(val_list)


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
#print(bin_val)
path = (r'C:\Users\Bendik Selvaag-Hagen\OneDrive - Universitetet i Oslo\Documents\GitHub\Master\numpy_data.csv')

end = time.time()
np.save(path, bin_val)

    

    
print('Seconds:',end-start)
  

