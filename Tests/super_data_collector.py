# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:06:02 2024

@author: Bendik Selvaag-Hagen
"""
import sys
sys.path.append(r'C:\Users\Bendik Selvaag-Hagen\OneDrive - Universitetet i Oslo\Documents\GitHub\Master')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Super_Data:
    def __init__(self, dim, problem_func, log_thresh):
        self.problem_func   = problem_func
        self.hist_data      = []
        self.dim            = dim
        self.hist_data      = []
        self.nfe            = 0
        self.best           = 10
        self.log_thresh     = log_thresh
        self.Data    = Problem_Function(self.dim)
        self.Data.param_change(0, self.log_thresh)        
        #self.open_data()
        
    def initialize_population(self, xmin, xmax, num_ind):
        self.xmin,self.xmax     = xmin,xmax
        self.num_ind            = num_ind
        self.individual         = np.ones((self.num_ind, self.dim))
        self.likelihood         = np.ones((self.num_ind))
        self.actual_likelihood  = np.ones_like(self.likelihood)
        self.conv_iter          = 0

        for i in range(self.num_ind):
            var     = []
            for j in range(self.dim):
                self.individual[i,j] = np.random.uniform(xmin, xmax)
                var.append(self.individual[i,j])
            temp, z = self.Data.evaluate(self.individual[i,:], self.problem_func)
            self.likelihood[i] = temp
            var.append(z)
            self.hist_data.append(var)
        self.nfe += self.num_ind
        self.open_data()
        self.hist_data = []


    def evolve(self, max_nfe):
        self.v          = np.zeros_like(self.individual)
        sort            = np.argsort(self.likelihood, axis = 0)
        F = 0.1
        #self.abs_best   = self.likelihood[best_indexes[0]]
        #self.best_ind   = self.individual[best_indexes][0]
        
        for _ in range(max_nfe):
            for i in range(self.num_ind):
                ri1 = np.random.randint(0,self.num_ind)
                ri2 = np.random.randint(0,self.num_ind)
                ri3 = np.random.randint(0,self.num_ind)
                
                #rand/1/bin
                self.v[i] = self.individual[ri1] + F*(self.individual[ri2] - self.individual[ri3] )
    
            for i in range(self.num_ind):
                perceived, true = self.Data.evaluate(self.v[i], self.problem_func)
                self.nfe += 1
                var = []
                for j in range(self.dim):
                    var.append(self.v[i,j])
                var.append(true)
                self.hist_data.append(var)
                
                u =  np.random.uniform(0,1)
                if perceived < self.likelihood[i]:
                    if np.random.uniform(0,1)  < 0.8:
                        self.individual[i] = self.v[i]
                        self.likelihood[i] = perceived
                        
                elif u  < 0.5:
                    self.individual[i] = self.v[i]
                    self.likelihood[i] = perceived
                if np.random.uniform(0,1) < 0.01:
                    for i in range(self.dim):
                        self.individual[i,j] += np.random.uniform(-1,1) 
                    perceived, true = self.Data.evaluate(self.v[i], self.problem_func)
                    self.nfe += 1
                    var = []
                    for j in range(self.dim):
                        var.append(self.v[i,j])
                    var.append(true)
                    self.hist_data.append(var)
                    self.likelihood[i] = perceived
            if len(self.hist_data)>int(1e5):
                print('nfe:',self.nfe, 'hist_data:', len(self.hist_data))
                self.write_data()
            
            if self.nfe > max_nfe:
                self.write_data()
                break
            
    def open_data(self):
        self.path = (r'C:\Users\Bendik Selvaag-Hagen\OneDrive - Universitetet i Oslo\Documents\GitHub\Master\datafile.csv')
        #self.path = (r"C:\Users\Lenovo\Documents\Master\datafile.csv")
        with open(self.path, 'w', newline='') as csvfile:
            csvfile = csv.writer(csvfile, delimiter=',')
            csvfile.writerows(self.hist_data)
            
    def write_data(self):
        # print('Length of written datafile:',len(self.hist_data))
        with open(self.path, 'a', newline='') as csvfile:
            csvfile = csv.writer(csvfile, delimiter=',')
            csvfile.writerows(self.hist_data)
        self.hist_data       = []


    def extract_data(self):
        #path    = (r"C:\Users\Lenovo\Documents\Master\datafile.csv")
        path    = (r'C:\Users\Bendik Selvaag-Hagen\OneDrive - Universitetet i Oslo\Documents\GitHub\Master\datafile.csv')
        ds      = pd.read_csv(path, delimiter=',',header = None)#, on_bad_lines='skip')
        
        likelihood_list = []
        individual_list = []
        #index_list      = []
        for index, row in ds.iterrows(): 
            var = []
            for j in range(self.dim):     
                var.append(row[j])
            likelihood_list.append(row[j+1])
            individual_list.append(var)
            #index_list.append(row[j+2])
        self.likelihood     = np.array(likelihood_list)
        self.individuals    = np.array(individual_list)
        #self.index          = np.array(index_list)
        a = np.where(self.likelihood<3.09)
        self.likelihood = self.likelihood[a]
        self.individuals = self.individuals[a,:]
        #print(len(self.likelihood))
        
    def visualize_parameter_space(self):
        #Sort order of the individuals 
        print(len(self.likelihood))
        print(np.shape(self.individuals[0]))
        self.individuals = self.individuals[0]
        sort_               = np.argsort(((-1)*self.likelihood), axis=0)
        self.likelihood     = self.likelihood[sort_]
        self.individuals    = self.individuals[sort_,:].reshape((len(self.likelihood), self.dim))
                
                
        fontsize = 8
        fsize_per_dim = 5.0
        fsize = fsize_per_dim * (self.dim-1)
        markersize = 10.0 / (self.dim-1)
        markerborderwidth = 0.25 / (self.dim-1)
        markerbordercolor = '0.3'
        figpad = 0.9 / fsize  # 0.07
        plotpad = 0.9 / fsize # 0.07
        plot_width = ( 1.0 - 2*figpad - max(0,self.dim-2)*plotpad ) / (self.dim - 1.)
        plot_height = plot_width

        fig = plt.figure(figsize=(fsize, fsize))

        for i in range(self.dim):
            for j in range(i+1,self.dim):
            # Add axes
                cmap = mpl.cm.viridis_r
                bounds = [0, 1.15, 3.09, 5.915]
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N) #extend = 'both'
                #cmap.set_extremes(under= 'black', over='black')
                left = figpad + i*(plot_width + plotpad)
                bottom = figpad + (j-1)*(plot_width + plotpad)
                ax = fig.add_axes((left, bottom, plot_width, plot_height))
                ax.set_facecolor('0')

                # Axis ranges
                
                
                plt.xlim([1.01*self.xmin, 1.01*self.xmax])
                plt.ylim([1.01*self.xmin, 1.01*self.xmax])


                #plt.xlim([-5,2])
                #plt.ylim([0,2])

                # Axis labels
                plt.ylabel("x_" + str(j), fontsize=fontsize)
                plt.xlabel("x_" + str(i), fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)

                scat = ax.scatter(self.individuals[:,i], self.individuals[:,j], c=self.likelihood,
                                    s=markersize, edgecolor=markerbordercolor, linewidth=markerborderwidth, cmap=cmap, norm=norm)

                # Add a colorbar for last plot on each row
                if j == i+1:
                    axins = inset_axes(ax, width="3%", height="100%", loc='right', borderpad=-0.4*fsize*plot_width)
                    cbar = fig.colorbar(scat, cax=axins, orientation="vertical")
                    cbar.set_label("objfunc(x)", fontsize=fontsize)
        ax = fig.add_axes((0.7 ,0.08, .25,.25))        
        counts, bins, patches = plt.hist(self.likelihood, bins=80, range=(0, 20), density=False, facecolor="forestgreen", alpha=0.5, histtype="stepfilled")
        max_counts = np.max(counts)

        plt.plot([1.15, 1.15], [0, 2 * max_counts], '--', color='black', label = '1sigma')
        plt.plot([3.09, 3.09], [0, 2 * max_counts], '--', color='grey', label = '2sigma')
        plt.plot([5.915, 5.915], [0, 2 * max_counts], '--', color='purple', label = '3sigma')
        plt.legend(loc = 0, bbox_to_anchor=(0.5, 1.1))
        plt.xlabel('log-likelihood')
        plt.ylabel('counts')
        plt.xlim([-0.2, 20.])
        plt.ylim([0., 1.1 * max_counts])

        summary_string = f"Data collector on Function: {self.problem_func}"
        fig = plt.gcf()
        plt.text(0.5, 1.0, summary_string, fontsize=fontsize, transform=fig.transFigure,
                verticalalignment='top', horizontalalignment='center')
        #plot_file_name = r"C:\Users\Lenovo\Documents\Master\Figs\fig_0602.png"
        #plt.savefig(plot_file_name)
        plt.show()


dim = 3
problem_func = 'mod_Himmelblau'
log_thresh = 3.09
data_collector = Super_Data(dim, problem_func, log_thresh)

xmin, xmax = -5,5
num_ind = 20000

data_collector.initialize_population(xmin, xmax, num_ind)
data_collector.evolve(int(1e7))
data_collector.extract_data()
data_collector.visualize_parameter_space()