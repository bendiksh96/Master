# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 11:03:53 2024

@author: Bendik Selvaag-Hagen
"""
import sys
sys.path.append(r"C:\Users\Lenovo\Documents\GitHub\Master\Physics\Methods")
from jde import *
from scipy.interpolate import LinearNDInterpolator
import numpy as np
import math
import pandas as pd


path_acc = r"C:\Users\Lenovo\Documents\GitHub\Master\Physics\HEPdata\Sig_acc_1.csv"
path_eff = r"C:\Users\Lenovo\Documents\GitHub\Master\Physics\HEPdata\Sig_eff_1.csv"
xa, ya, za = np.loadtxt(path_acc, comments='#', delimiter=',', unpack=True)
xe, ye, ze = np.loadtxt(path_eff, comments='#', delimiter=',', unpack=True)

def acc_eff(ind):    
    interp_acc = LinearNDInterpolator(list(zip(xa, ya)), za, fill_value=np.nan)
    interp_eff = LinearNDInterpolator(list(zip(xe, ye)), ze, fill_value=np.nan)
    acc = interp_acc.__call__(ind[0], ind[1])
    eff = interp_eff.__call__(ind[0], ind[1])
    return acc, eff


def eval_likelihood(x):
    func = 0
    #Kriteriers
    if x[0] < x[2]:
        func += 100
    if x[0] > x[1]:
        func += 100
    return func

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

    acc, eff = acc_eff(x)
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








