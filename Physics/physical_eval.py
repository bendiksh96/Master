from scipy.interpolate import LinearNDInterpolator
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import xsec

class XSection:
    def __init__(self, ggd_list):    
        xsec.init(data_dir="gprocs")
        xsec.set_energy(13000)
        processes = [(1000021, 1000021)]
        xsec.load_processes(processes)
        
        #Of type ['1','2','3','4']
        self.ggd_list = ggd_list
        
        
        
        path_acc = r"C:\Users\Lenovo\Documents\GitHub\Master\Physics\HEPdata\Sig_acc_1.csv"
        path_eff = r"C:\Users\Lenovo\Documents\GitHub\Master\Physics\HEPdata\Sig_eff_1.csv"
        xa, ya, za = np.loadtxt(path_acc, comments='#', delimiter=',', unpack=True)
        xe, ye, ze = np.loadtxt(path_eff, comments='#', delimiter=',', unpack=True)
        self.interp_acc_1 = LinearNDInterpolator(list(zip(xa, ya)), za, fill_value=np.nan)
        self.interp_eff_1 = LinearNDInterpolator(list(zip(xe, ye)), ze, fill_value=np.nan)
        
        
        if len(ggd_list) == 1:
            self.n_1 = 45 
            self.b_1 = 30
            self.fac_n_1 = float(math.factorial(self.n_1))
            self.background_1 = self.n_1 * np.log(self.b_1) - self.b_1 - np.log(self.fac_n_1)
        
        elif len(ggd_list) == 2:
            self.n_2 = 68 
            self.b_2 = 52
            self.fac_n_2 = float(math.factorial(self.n_2))
            self.background_2 = self.n_2 * np.log(self.b_2) - self.b_2 - np.log(self.fac_n_2)
        
        elif len(ggd_list) == 3:
            self.n_3 = 227 
            self.b_3 = 223
            self.fac_n_3 = float(math.factorial(self.n_3))
            self.background_3 = self.n_3 * np.log(self.b_3) - self.b_3 - np.log(self.fac_n_3)
        
        elif len(ggd_list) == 4:
            self.n_4 = 291
            self.b_4 = 298
            self.fac_n_4 = float(math.factorial(self.n_4))
            self.background_4 = self.n_4 * np.log(self.b_4) - self.b_4 - np.log(self.fac_n_4)
        
        self.modifier = 0
        self.abs_best = 0
        self.conv_    = False

    def acc_eff(self,ind): 
        acc = self.interp_acc_1.__call__(ind[0], ind[1])
        eff = self.interp_eff_1.__call__(ind[0], ind[1])
        acc = acc/100; eff = eff/100
        return acc, eff

    def eval_cross_section(self, ind):
        gluino_mass = ind[0]
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
        signal_status = True
        L = 139 #fb^-1
        acc, eff = self.acc_eff(ind)
        if np.isnan(acc) or np.isnan(eff):
            s = -1
            signal_status = False
            return s, signal_status, np.nan
        
        sigma_gg = self.eval_cross_section(ind)
        s = L * sigma_gg[0] * acc * eff
        return s, signal_status, sigma_gg
    
    def evaluate(self, ind):
        """
        Parameters
        ----------
        ind : individual
        Returns
        -------
        func : target function x2
        """
        func_crit, crit_status = self.criteria(ind)
        if crit_status == False:
            return func_crit, func_crit, np.nan, np.nan
        
        signal, signal_status, section = self.signal_prediction(ind)
        if signal_status == False:
            tramp = 1e4
            return tramp, tramp, np.nan, np.nan
        
        pred    = self.n_1*np.log(signal+self.b_1) - (signal + self.b_1) - np.log(self.fac_n_1)
        pred_true = pred
        if pred < self.abs_best + self.modifier and self.conv_ == True:
            pred = self.modifier + (self.modifier- self.abs_best)
        target_true = -(pred_true - self.background_1)
        target  = -( pred - self.background_1)
        return target, target_true, signal, section
    
    def criteria(self, ind):
        """_summary_
        Checks the criteria of the parameters. 
        If not met, set likelihood. Criter
        Args:
            ind (_type_): individual

        Returns:
            func: high value
            status (_bool_): True if criteria check is good, False if not.
        """
        func = 0
        status = True
        #Gluino mass must be smaller than squark mass
        if ind[0] > ind[2]:
            func = 100 * ( ind[0] - ind[2])
            status = False
            
        #Gluino mass must be larger than neutralino mass
        if ind[0] < ind[1]:
            func = 100 * (ind[1] - ind[0])
            status = False
            
        return func, status



