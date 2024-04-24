# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 11:03:53 2024

@author: Bendik Selvaag-Hagen
"""
import sys
import csv
import numpy as np

sys.path.append(r"C:\Users\Lenovo\Documents\GitHub\Master\Physics\Methods")
from jde import *


class Physics:
    def __init__(self, method):
        self.method = method
        self.dim = 3 
        #For BDT
        self.n = 45
        self.num_ind = 1000

        self.nfe = 0 ; self.max_nfe = 5e4

        #Observed events,Total bkg post-fit, new estimates
        self.b_d1 = 30
        self.b_d2 = 52
    def evolution(self):
        if self.method == 'jde':
            mod = jDE(self.num_ind)   
            mod.initialize_population()  
            self.open_data(mod.hist_data)
            mod.hist_data = []
                       
            while self.nfe < self.max_nfe:
                mod.evolve()
                self.nfe  = mod.nfe
                print('nfe:',self.nfe)

                if len(mod.hist_data)>int(1e5):
                    self.write_data(mod.hist_data)
                    mod.hist_data       = []
            self.write_data(mod.hist_data)
                
                
        if self.method == 'shade':
            mod = SHADE(self.num_ind)   
            mod.initialize_population()  
            self.open_data(mod.hist_data)
            mod.hist_data = []
                       
            while self.nfe < self.max_nfe:
                mod.evolve()
                self.nfe  = mod.nfe
                print('nfe:',self.nfe)

                if len(mod.hist_data)>int(1e5):
                    self.write_data(mod.hist_data)
                    mod.hist_data       = []
            self.write_data(mod.hist_data)
                
                
                
        if self.method == 'double_shade':
            while nfe < max_nfe:
                arg = 0
                
                
        if self.method == 'double_shade_pso':
            while nfe < max_nfe:
                arg = 0
                
    def open_data(self,data):
        self.path = r"C:\Users\Lenovo\Documents\GitHub\Master\Physics\eval_data.csv"
        with open(self.path, 'w', newline='') as csvfile:
            csvfile = csv.writer(csvfile, delimiter=',')
            csvfile.writerows(data)

    def write_data(self, data):
        with open(self.path, 'a', newline='') as csvfile:
            csvfile = csv.writer(csvfile, delimiter=',')
            csvfile.writerows(data)


method_list = ['jde', 'shade']

a = Physics(method_list[0])
a.evolution()





