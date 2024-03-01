# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 10:29:36 2024

@author: Bendik Selvaag-Hagen
"""

import numpy as np
import matplotlib.pyplot as plt

def func(x,y):
    return x**2 +    y*2


def ripley(ind, lik, t):
    nur = 100
    x = np.linspace(-2,2,nur)
    y = np.linspace(-2,2,nur)
    
    #X, Y =  np.meshgrid(x,y)
    #likelihood_grid = func(X,Y)
    
    dist = np.zeros((nur,nur))
    val  = np.zeros((nur,nur))
    I = np.zeros_like(dist)
    k = 11
    #for k in range(num_ind):
    for i in range(len(x)):
        for j in range(len(y)):
            dist[i,j] = np.sqrt( (x[i]-ind[k,0])**2 + (y[i]-ind[k, 1])**2)
            if dist[i,j] < t:
       #         print(dist[i,j], t)
                I[i,j] = 1
                print(k)
            else:
                I[i,j] = 'nan'
                #I[i,j] = abs(lik[k] - likelihood_grid[i,j])  
    #min_ind = np.where(dist < t)
    
    #I[min_ind] = 1
    
    print(I)
    print(np.where(I==1))
    return I
    
num_ind = 20
dim     = 2
individual = np.zeros((num_ind, dim))
likelihood = np.zeros(num_ind)
xmin,xmax = -2,2 

for i in range(num_ind):
    for j in range(dim):
        individual[i,j] = np.random.uniform(xmin, xmax)
    likelihood[i] = func(individual[i,0], individual[i,1])
    
F = 0.1
CR = 0.6
u = np.zeros_like(individual)
for _ in range(10):
    #sort = np.argsort(likelihood)
    for i in range(num_ind):
        r1 = np.random.randint(num_ind)
        r2 = np.random.randint(num_ind)
        r3 = np.random.randint(num_ind)
        u[i] = individual[r1] + F*(individual[r2] -  individual[r3])
    
    for i in range(num_ind):
        ru = np.random.uniform(0,1)
        if CR < ru:
            l = func(u[i,0], u[i,1])
            if l < likelihood[i]:
                individual[i] = u[i]
                likelihood[i] = l

tur = 100
np.random.seed(2121313)
x = np.linspace(-2,2,tur)
y = np.linspace(-2,2,tur)
X,Y = np.meshgrid(x,y)
t = np.linspace(0.7,1,9)
val = []
val2 = []
fig, axs = plt.subplots(3,3, figsize = (10,10)) 
count1 = 0; count2 = 0
for t_ in t:
    f = ripley(individual, likelihood, t_)
    #print(np.where(f==1))
    val.append(f)
   
    val2.append(f-t_)
    print(f)
    print(np.where(f==1))
    axs[count1,count2].contourf(X,Y, f)
    
    break
    count1 += 1
    
    if count1 >2:
        count1 = 0
        count2+=1

