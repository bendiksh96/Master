import os
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import seaborn as sns


class Vis:
    def __init__(self, dim, x_min, x_max, nfe_max, method, problem_function):
        self.x_min, self.x_max  = x_min, x_max
        self.dim                = dim
        self.nfe_max            = nfe_max
        self.method             = method
        self.problem_function   = problem_function
        

    def extract_data(self):
        path    = (r"C:\Users\Lenovo\Documents\Master\datafile.csv")
        # path    = (r'C:\Users\Bendik Selvaag-Hagen\OneDrive - Universitetet i Oslo\Documents\GitHub\Master\datafile.csv')
        ds      = pd.read_csv(path, delimiter=',',header = None)#, on_bad_lines='skip')
        
        likelihood_list = []
        individual_list = []
        index_list      = []
        for index, row in ds.iterrows(): 
            var = []
            for j in range(self.dim):     
                var.append(row[j])
            likelihood_list.append(row[j+1])
            individual_list.append(var)
            index_list.append(row[j+2])
            
        self.likelihood     = np.array(likelihood_list)
        self.individuals    = np.array(individual_list)
        self.index          = np.array(index_list)
        print('Number of points:', len(self.likelihood))
        
        
        
    def visualize_parameter_space(self):
        #Sort order of the individuals 
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
                plt.xlim([1.01*self.x_min, 1.01*self.x_max])
                plt.ylim([1.01*self.x_min, 1.01*self.x_max])

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
        # l = np.where(self.likelihood < 10)

        # lik = self.likelihood[l]
        
        # dex = self.index[l]
        # fontsize = 8
        # labels = sorted(set(dex))
        # bin = 80
        # colors = plt.cm.get_cmap('tab10', len(labels))
        # colors = sns.color_palette("husl", len(labels))
        # counts = []
        
        # count,b = np.histogram(lik)
        # jar = []
        # for i, method in enumerate(labels):
        #     method_values = lik[dex == method]
        #     hist,_ = np.histogram(method_values, bins=bin)
        #     counts.append(hist)
        #     jar.append(method_values)
        # max_count = np.max(counts)*2
        
        # sns.histplot(jar,multiple = 'stack', bins = bin) 
        # sns.lineplot(x=[1.15, 1.15], y=[0, max_count*1.1], dashes=[3,3])                        
        # sns.lineplot(x=[3.09, 3.09], y=[0, max_count*1.1], dashes=[3,3])                        
        # sns.lineplot(x=[5.915, 5.915], y=[0, max_count*1.1], dashes=[3,3])                        




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

        summary_string = f"Method: {self.method} on Function: {self.problem_function}"
        fig = plt.gcf()
        plt.text(0.5, 1.0, summary_string, fontsize=fontsize, transform=fig.transFigure,
                verticalalignment='top', horizontalalignment='center')
        plot_file_name = r"C:\Users\Lenovo\Documents\Master\Figs\fig_0602.png"
        plt.savefig(plot_file_name)
        plt.show()


    def stacked_hist(self):
        l = np.where(self.likelihood < 10)
        self.likelihood = self.likelihood[l]
        
        self.index = self.index[l]
        fontsize = 8
        labels = sorted(set(self.index))
        bin = 80
        # colors = plt.cm.get_cmap('tab10', len(labels))
        # colors = sns.color_palette("husl", len(labels))
        # count,b = np.histogram(self.likelihood)
        counts = []
        
        jar = []
        for i, method in enumerate(labels):
            method_values = self.likelihood[self.index == method]
            hist,_ = np.histogram(method_values, bins=bin)
            counts.append(hist)
            jar.append(method_values)
        max_count = np.max(counts)
        
        sns.histplot(jar,multiple = 'stack', bins = bin) 
        sns.lineplot(x=[1.15, 1.15], y=[0, max_count*1.1], dashes=[3,3])                        
        sns.lineplot(x=[3.09, 3.09], y=[0, max_count*1.1], dashes=[3,3])                        
        sns.lineplot(x=[5.915, 5.915], y=[0, max_count*1.1], dashes=[3,3])                        
        plt.show()
        
        
    def visualize_2(self):#, iter_likelihood_mean, iter_likelihood_min,iter_likelihood_median):
        # Create figure
        fsize_per_dim = 5.0
        fsize = fsize_per_dim * (self.dim-1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3 * fsize_per_dim, fsize_per_dim))

        # Plot 1: Histogram of objfunc_vals
        plt.sca(ax1)

        # Axis labels
        plt.ylabel("Number of points")
        plt.xlabel("-loglike")

        # Create plot
        counts, bins, patches = plt.hist(self.likelihood, bins=80, range=(0, 20.), density=False, facecolor="forestgreen", alpha=0.5, histtype="stepfilled")

        max_counts = np.max(counts)

        plt.plot([1.15, 1.15], [0, 2 * max_counts], '--', color='black')
        plt.plot([3.09, 3.09], [0, 2 * max_counts], '--', color='black')
        plt.plot([5.915,5.915], [0, 2 * max_counts], '--', color='black')

        plt.xlim([-0.2, 8.])
        plt.ylim([0., 1.1 * max_counts])

        #delta_x = x_max - x_min
        fig = plt.gcf()
        plt.text(0.5, 1.0, f"{self.method} on {self.problem_function}", fontsize=8.0, transform=fig.transFigure,
                verticalalignment='top', horizontalalignment='center')
        # Plot 2: objective function values vs iteration
        plt.sca(ax2)

        iterations = list(range(len(iter_likelihood_mean)))
        plt.plot(iterations, iter_likelihood_mean, '-', color = 'black', label = 'Mean likelihood')
        plt.plot(iterations, iter_likelihood_min, '-', linewidth = 1.5, label = 'Best likelihood')
        plt.plot(iterations, iter_likelihood_median, '--', linewidth =1.5, label = 'Median likelihood')
        plt.xlabel("Iteration")
        plt.ylabel("-loglike")
        plt.yscale("log")
        plt.ylim([1e0, 1e1])
        plt.legend()
        plt.show()
        # plot_file_name = os.path.join(output_dir_name, "summary_plots.png")
        # plt.savefig(plot_file_name)

        # print()
        # print(f"Generated plot: {plot_file_name}")

    def visualize_population_evolution(self, individuals, likelihood):
        # Create figure
        fontsize = 8
        fsize = 14
        markersize = 10.0 / (self.dim-1)
        markerborderwidth = 0.25 / (self.dim-1)
        markerbordercolor = '0.3'
        figpad = 1 #0.9 / fsize  # 0.07
        plotpad = 1 #0.9 / fsize # 0.07
        plot_width = fsize/90# 1.0 - 2*figpad - max(0,self.dim-2)*plotpad ) / (self.dim - 1.)
        plot_height = plot_width
        left = 0.1#figpad + (plot_width + plotpad)
        bottom = 0.07#figpad + 0.3*(plot_width + plotpad)
        
        fig = plt.figure(figsize=(fsize, fsize))
        cmap_vmin = 0.0
        cmap_vmax = 6
        plot_facecolor = '0.0'

        count = 0
        max_count = 3
        initial = 0
        ax = fig.add_axes((left, bottom, plot_width, plot_height))
        ax.set_title(f'Total individual distirbution after {0} - {max_count+1} iterations',fontsize=' 6')
        for iter in range(np.size(individuals, axis=0)):
            # Axis ranges
            plt.xlim([1.05*self.x_min, 1.05*self.x_max])
            plt.ylim([1.05*self.x_min, 1.05*self.x_max])
            # Axis labels
            plt.ylabel("x_1" , fontsize=fontsize)
            plt.xlabel("x_0" , fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            ax.set_facecolor(plot_facecolor)
            ax.scatter(individuals[iter, :,0], individuals[iter, :,1], s=markersize)#, c=likelihood,
                                # s=markersize, edgecolor=markerbordercolor, linewidth=markerborderwidth, cmap=cmap, norm=norm)
            if count > max_count:
                left += 0.2 
                if left > 0.8:
                    bottom += 0.24
                    left  = 0.1
                if bottom >0.8:
                    plot_name = f'Iter_{initial}_to_{count}.jpg'
                    fig.savefig(plot_name)
                    plt.show()
                    bottom = 0.05
                    fig = plt.figure(figsize=(fsize, fsize))

                ax = fig.add_axes((left, bottom , plot_width, plot_height))
                ax.set_title(f'Total individual distirbution after {iter} - {iter+max_count+1} iterations',fontsize=' 6')
                count = 0
            count +=1

        plt.show()            
        


