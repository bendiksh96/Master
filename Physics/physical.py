# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 11:03:53 2024

@author: Bendik Selvaag-Hagen
"""
import sys
import csv
import numpy as np

sys.path.append(r"C:/Users/Lenovo/Documents/GitHub/Master/Physics/Methods")
from jde import *
from shade import *
from dshade import *
from dshadepso  import *


class Physics:
    def __init__(self, method, num_ind, max_nfe, ggd_list):
        self.method = method
        self.dim = 3 
        
        self.ggd_list = ggd_list
        
        
        #For BDT        
        self.num_ind = num_ind

        self.nfe = 0 ; self.max_nfe = max_nfe

        #Observed events,Total bkg post-fit, new estimates
        
    def evolution(self):
        if self.method == 'jde':
            mod = jDE(self.num_ind, self.ggd_list)   
            print('Particles initialized')

            mod.initialize_population()  
            self.open_data(mod.hist_data)
            mod.hist_data = []
            while self.nfe < self.max_nfe:
                mod.evolve()
                self.nfe  = mod.nfe
                if len(mod.hist_data)>int(1e4):
                    print('nfe:',self.nfe)                    
                    self.write_data(mod.hist_data)
                    mod.hist_data       = []
            self.write_data(mod.hist_data)
                             
        if self.method == 'shade':
            mod = SHADE(self.num_ind, self.ggd_list)   
            mod.initialize_population()  
            self.open_data(mod.hist_data)
            mod.hist_data = []
            while self.nfe < self.max_nfe:
                mod.evolve()
                self.nfe  = mod.nfe

                if len(mod.hist_data)>int(1e5):
                    print('nfe:',self.nfe)
                    self.write_data(mod.hist_data)
                    mod.hist_data       = []
            self.write_data(mod.hist_data)

        if self.method == 'dshade':
            mod = dSHADE(self.num_ind, self.ggd_list)   
            mod.initialize_population()  
            print('Particles initialized')
            self.open_data(mod.hist_data)
            mod.hist_data = []
            conv  = False
            while self.nfe < self.max_nfe and conv == False:
                mod.evolve_converge()
                self.nfe  = mod.nfe
                if len(mod.hist_data)>int(1e3):
                    self.write_data(mod.hist_data)
                    mod.hist_data       = []
                    print('nfe:',self.nfe)
                    
                if mod.abs_best < -4.9:
                    conv = True
                    break
            self.write_data(mod.hist_data)
            mod.hist_data   = []
            
            print('Converged -- Exploring')
            mod.modifier    = 3.09  #2sigma
            self.num_ind = self.num_ind*100
            mod.__init__(self.num_ind, self.ggd_list)
            mod.alter_likelihood()
            mod.initialize_population()
            
            while self.nfe < self.max_nfe:
                mod.evolve_explore()
                self.nfe = mod.nfe
                if len(mod.hist_data)>int(1e4):
                    self.write_data(mod.hist_data)
                    mod.hist_data       = []
                    print('nfe:',self.nfe)
            
                
                
        if self.method == 'double_shade_pso':
            mod = dSHADEpso(self.num_ind, self.ggd_list)   
            mod.initialize_population()  
            print('Particles initialized')
            self.open_data(mod.hist_data)
            mod.hist_data = []
            conv  = False
            while self.nfe < self.max_nfe and conv == False:
                mod.evolve_converge()
                self.nfe  = mod.nfe
                if len(mod.hist_data)>int(1e3):
                    self.write_data(mod.hist_data)
                    mod.hist_data       = []
                    print('nfe:',self.nfe)
                    
                if mod.abs_best < -3.244:
                    conv = True
                    break
            self.write_data(mod.hist_data)
            mod.hist_data   = []
            
            print('Converged -- Exploring')
            mod.modifier    = 3.09  #2sigma
            self.num_ind = self.num_ind*100
            mod.__init__(self.num_ind, self.ggd_list)
            mod.alter_likelihood()
            mod.initialize_population()
            
            while self.nfe < self.max_nfe:
                mod.evolve_explore()
                self.nfe = mod.nfe
                if len(mod.hist_data)>int(1e4):
                    self.write_data(mod.hist_data)
                    mod.hist_data       = []
                    print('nfe:',self.nfe)
                if self.nfe/self.max_nfe == 0.8:
                    break
            centroids, clusters = mod.cluster(0, k=20)
            mod.optimum = mod.abs_best 
            for i in range(self.num_ind):
                arg = int(clusters[i])
                mod.optimal_individual[i] = centroids[arg]
            mod.init_particles()
            while self.nfe < self.max_nfe:
                mod.evolve_particle()
                self.nfe = mod.nfe
                if len(mod.hist_data)>int(1e4):
                    self.write_data(mod.hist_data)
                    mod.hist_data       = []
                    print('nfe:',self.nfe)
            self.write_data(mod.hist_data)
            
                
                
    def open_data(self,data):
        self.path = r"C:/Users/Lenovo/Documents/GitHub/Master/Physics/dshadepso.csv"
        with open(self.path, 'w', newline='') as csvfile:
            csvfile = csv.writer(csvfile, delimiter=',')
            csvfile.writerows(data)

    def write_data(self, data):
        with open(self.path, 'a', newline='') as csvfile:
            csvfile = csv.writer(csvfile, delimiter=',')
            csvfile.writerows(data)


method_list = ['jde', 'shade', 'dshade', 'double_shade_pso']

a = Physics(method_list[3], num_ind = 100, max_nfe=1e5, ggd_list=['1'])#, '2', '3', '4'])
a.evolution()






