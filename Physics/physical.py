# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 11:03:53 2024

@author: Bendik Selvaag-Hagen
"""

#Forsøk på fysisk problemstilling
import sys
sys.path.append(r'C:\Users\Bendik Selvaag-Hagen\OneDrive - Universitetet i Oslo\Documents\GitHub\Master\Physics\Methods')
#from jde import *


import numpy as np
import math
import pandas as pd


"""
from bat import *
from ddms import * 
from shade import *
from shabat import *
from jderpo import *
from d_shade import *
from shade_bat import *
from chaotic_bat import *
from d_shade_pso import * 
from d_shade_bat import *
"""
def eval_likelihood(x):
    
    func = 0
    #Kriteriers
    if x[0] < x[2]:
        func += 100
    if x[0] > x[1]:
        func += 100
    return func



num_g   = 25
num_xi  = 40
g_distribution = np.linspace(500, 3000, num_g+1)
xi_distribution = np.linspace(0, 2000, num_xi+1)

def acc_reader():
    acc_arr = np.ones((num_g+1, num_xi+1))*(-1  )
    
    if d1:
        path_acc = r"C:\Users\Bendik Selvaag-Hagen\OneDrive - Universitetet i Oslo\Documents\GitHub\Master\Physics\HEPdata\Sig_acc_1.csv"
    if d2:
        path_acc = r"C:\Users\Bendik Selvaag-Hagen\OneDrive - Universitetet i Oslo\Documents\GitHub\Master\Physics\HEPdata\Sig_acc_1.csv"
    ds = pd.read_csv(path_acc)
    for index, row in ds.iterrows():
        x_axis = np.where(g_distribution == row[0])
        y_axis = np.where(xi_distribution == row[1])
        acc_arr[x_axis, y_axis] = row[2]
    return acc_arr

def eff_reader():
    """
    Returns
    -------
    eff_arr : Array of type (g, xi) with values of efficiency

    """
    eff_arr = np.ones((num_g+1, num_xi+1))*(-1)
    if d1:
        path_eff = r"C:\Users\Bendik Selvaag-Hagen\OneDrive - Universitetet i Oslo\Documents\GitHub\Master\Physics\HEPdata\Sig_eff_1.csv"
    if d2:
        path_eff = r"C:\Users\Bendik Selvaag-Hagen\OneDrive - Universitetet i Oslo\Documents\GitHub\Master\Physcs\HEPdata\Sig_eff_2.csv"
    ds = pd.read_csv(path_eff)
    
    for index, row in ds.iterrows():
        x_axis = np.where(g_distribution == row[0])
        y_axis = np.where(xi_distribution == row[1])
        eff_arr[x_axis, y_axis] = row[2]
    return eff_arr


d1 = 1
d2 = 0
acc_arr = acc_reader()
eff_list = eff_reader()

    
def acc_eff_extract(individual):
    acc, eff = -1,-1
    for i in range(len(g_distribution-1)):
        for j in range(len(xi_distribution)):
            if g_distribution[i] < individual[0] < g_distribution[i+1] and g_distribution[i] < individual[1] < g_distribution[i+1]:
                acc = acc_array[i,j]
                eff = eff_array[i,j]
    return acc, eff

def sigma_extract(x):
    """
    Parameters
    ----------
    x : Individual.

    Returns crossection
    -------
    
    Call on xsec to create a cross section with given parameters. 
    """
    return 1
    
    

def signal_prediction(x):
    """

    Parameters
    ----------
    x : individual 
    sigma_gg : cross section

    Returns
    -------
    s : signal prediction

    """
    
    
    
    L = 139 #fb^-1

    acc, eff = acc_eff_extract(x)
    sigma_gg = sigma_extract(x)
    s = L * sigma_gg * acc * eff
    return s


def log_lik(s, b):
    """

    Parameters
    ----------
    s : signal prediction
    b : TYPE
        DESCRIPTION.

    Returns
    -------
    func : target function 

    """
    
    a = float(math.factorial(n))
    func = -n*np.log(s+b) + (s + b) + np.log(a)
    return func    




dim = 3 
#For BDT
n = 45
num_ind = 100



nfe = 0 ; max_nfe = 1e4
dim = 3


#Observed events,Total bkg post-fit, new estimates
b_d1 = 30
b_d2 = 52



method = 'jde'
def evolution():
    #g. n, q
    ind_ind = ['g', 'n' , 'q']
    individual = np.zeros((num_ind, dim))
    likelihood = np.zeros(num_ind)
    #Initialize 
    for p in range(num_ind ):
        for j in range(dim):
            if ind_ind[j] == 'g' or ind_ind[j] == 'q':
                individual[p, j] = np.random.uniform(500,3000)
            if ind_ind[j] == 'n':
                individual[p, j] = np.random.uniform(0  ,2000)
                
    if method == 'jde':
        while nfe < max_nfe:
            b = 1
            tol = 1e-3; conv = False
            mod = jDE(individual, likelihood, problem_func)            
            while nfe < max_nfe and conv == False:
                mod.evolve()
                nfe  += mod.nfe
                if int(b*1e4)<nfe:
                    print('nfe:',nfe)
                    b+=1

                if len(hist_data)>int(1e5):
                    write_data(mod.hist_data)
                    mod.hist_data       = []
            write_data(mod.hist_data)
            
            
    if method == 'shade':
        while nfe < max_nfe:
            arg = 0
            
            
    if method == 'double_shade':
        while nfe < max_nfe:
            arg = 0
            
            
    if method == 'double_shade_pso':
        while nfe < max_nfe:
            arg = 0
    
    sig = signal_prediction(individual[5])
    target = log_lik(sig, b_d1)
    print(target)








