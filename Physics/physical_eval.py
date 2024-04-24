from scipy.interpolate import LinearNDInterpolator
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import xsec

class XSection:
    def __init__(self):    
        xsec.init(data_dir="gprocs")
        xsec.set_energy(13000)
        # Load GP models for the specified process gg-> n
        processes = [(1000021, 1000021)]
        xsec.load_processes(processes)
        
        path_acc = r"C:\Users\Lenovo\Documents\GitHub\Master\Physics\HEPdata\Sig_acc_1.csv"
        path_eff = r"C:\Users\Lenovo\Documents\GitHub\Master\Physics\HEPdata\Sig_eff_1.csv"
        xa, ya, za = np.loadtxt(path_acc, comments='#', delimiter=',', unpack=True)
        xe, ye, ze = np.loadtxt(path_eff, comments='#', delimiter=',', unpack=True)
        
        self.interp_acc = LinearNDInterpolator(list(zip(xa, ya)), za, fill_value=np.nan)
        self.interp_eff = LinearNDInterpolator(list(zip(xe, ye)), ze, fill_value=np.nan)

    def acc_eff(self,ind):    
        acc = self.interp_acc.__call__(ind[0], ind[1])
        eff = self.interp_eff.__call__(ind[0], ind[1])
        return acc, eff

    def eval_cross_section(self, ind):
        gluino_mass = ind[0]
        neutralino_mass = ind[1]
        squark_mass = ind[2]
        
        xsec.set_all_squark_masses(squark_mass)
        xsec.set_gluino_mass(gluino_mass)
        xsec.get_parameters()
        res = xsec.eval_xsection(verbose = 0)
        xsec.clear_parameter(gluino_mass)
        xsec.clear_parameter(squark_mass)
        return res[0]
    
    def signal_prediction(self,ind):
        """
        Parameters
        ----------
        ind : individual 

        Returns
        -------
        s : signal prediction

        """    
        L = 139 #fb^-1

        acc, eff = self.acc_eff(ind)
        sigma_gg = self.eval_cross_section(ind)
        if np.isnan(acc):
            acc = 100
        if np.isnan(eff):
            eff = 1000
        s = L * sigma_gg[0] * acc * eff
        # print('signal:',s, 'acceptance:',acc, 'efficiency',eff, 'cross section:',sigma_gg[0])
        return s
    
    def evaluate(self, ind):
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
        signal = self.signal_prediction(ind)
        n = 45
        b = 30
        a = float(math.factorial(n))
        func = -n*np.log(signal+b) + (signal + b) + np.log(a)
        # print('val:', func)
        # print()
        return func, func





