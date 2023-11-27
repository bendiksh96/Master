from pathlib import Path
import os
import matplotlib
import csv
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

class Data:
    def __init__(self, dim, x_min, x_max):
        self.dim = dim
        self.x_min = x_min
        self.x_max = x_max

    def data_storage(self):
        a = 1

    def visualize_1(self, individuals, likelihood, eval_type, output_dir=str(Path.cwd())):
        sort_ = np.argsort((likelihood), axis=0)[::-1]
        likelihood = likelihood[sort_]
        individuals = individuals[sort_,:].reshape((len(likelihood), self.dim))
        # kn =  np.where(likelihood > 60)
        # individuals[kn,:] = 'nan'

        # _Anders
        print(f"DEBUG: len(individuals): {len(individuals)}")

        # keep_indices =  np.where(likelihood < 6.0)
        # individuals = individuals[keep_indices]
        # likelihood = likelihood[keep_indices]
        # print(f"DEBUG: len(individuals): {len(individuals)}")

        print(f"DEBUG: likelihood[-10:]: {likelihood[-10:]}")
        print(f"DEBUG: likelihood[0:10]: {likelihood[0:10]}")


        # Create figure
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
        cmap_vmin = 0.0
        cmap_vmax = 6.0
        plot_facecolor = '0.0'

        for i in range(self.dim):
            for j in range(i+1,self.dim):
            # Add axes
                cmap = plt.get_cmap("viridis_r")
                # _Anders
                cmap.set_extremes(under= 'red', over='grey')
                # norm = plt.Normalize(0, 5.915)
                norm = matplotlib.cm.colors.Normalize(vmin=cmap_vmin, vmax=cmap_vmax)
                left = figpad + i*(plot_width + plotpad)
                bottom = figpad + (j-1)*(plot_width + plotpad)
                ax = fig.add_axes((left, bottom, plot_width, plot_height))
                ax.set_facecolor(plot_facecolor)

                # Axis ranges
                plt.xlim([1.05*self.x_min, 1.05*self.x_max])
                plt.ylim([1.05*self.x_min, 1.05*self.x_max])

                # Axis labels
                plt.ylabel("x_" + str(j), fontsize=fontsize)
                plt.xlabel("x_" + str(i), fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                # Create a colour scale normalization
                # norm = matplotlib.cm.colors.Normalize(vmin=cmap_vmin, vmax=cmap_vmax)
                # _Anders
                # scat = ax.scatter(individuals[:,i], individuals[:,j], c=likelihood,
                #                     s=markersize, edgecolor=markerbordercolor, linewidth=markerborderwidth, cmap=cmap, norm=norm)
                scat = ax.scatter(individuals[:,i], individuals[:,j], c=likelihood,
                                    s=markersize, edgecolor=markerbordercolor, linewidth=markerborderwidth, cmap=cmap, norm=norm)

                # Add a colorbar for last plot on each row
                if j == i+1:
                    axins = inset_axes(ax, width="3%", height="100%", loc='right', borderpad=-0.4*fsize*plot_width)
                    cbar = fig.colorbar(scat, cax=axins, orientation="vertical")
                    cbar.set_label("objfunc(x)", fontsize=fontsize)

        summary_string = f"Method: {eval_type}"
        fig = plt.gcf()
        plt.text(0.5, 1.0, summary_string, fontsize=fontsize, transform=fig.transFigure,
                verticalalignment='top', horizontalalignment='center')
        plot_file_name = str(Path(output_dir) / "fig.png")
        plt.savefig(plot_file_name)
        # plt.show()

    def visualize_2(self, iter, individuals, likelihood):
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
        counts, bins, patches = plt.hist(likelihood, bins=80, range=(0, 20.), density=False, facecolor="forestgreen", alpha=0.5, histtype="stepfilled")
        counts, bins, patches = plt.hist(likelihood, bins=80, range=(0, 20.), density=False, color="black", histtype="step", linewidth=2.0)

        max_counts = np.max(counts)

        plt.plot([1.0, 1.0], [0, 2 * max_counts], '--', color='black')
        plt.plot([3.0, 3.0], [0, 2 * max_counts], '--', color='black')
        plt.plot([6.0, 6.0], [0, 2 * max_counts], '--', color='black')

        plt.xlim([-0.5, 20.])
        plt.ylim([0., 1.1 * max_counts])

        #delta_x = x_max - x_min
        fig = plt.gcf()
        plt.text(0.5, 1.0, "Juppsi", fontsize=8.0, transform=fig.transFigure,
                verticalalignment='top', horizontalalignment='center')
        plt.show()
        exit()
        # Plot 2: objective function values vs iteration
        plt.sca(ax2)

        best_obj_iter = solution[2]
        mean_obj_iter = solution[3]
        median_obj_iter = solution[4]
        iterations = list(range(len(best_obj_iter)))

        plt.plot(iterations, best_obj_iter, '-', color='black', linewidth=1.5, label="best")
        plt.plot(iterations, mean_obj_iter, '-', linewidth=1.5, label="mean")
        plt.plot(iterations, median_obj_iter, '-', linewidth=1.5, label="median")

        plt.xlabel("Iteration")
        plt.ylabel("-loglike")

        plt.yscale("log")
        plt.ylim([1e-6, 1e4])

        plt.legend()

        plot_file_name = os.path.join(output_dir, "summary_plots.png")
        plt.savefig(plot_file_name)

        print()
        print(f"Generated plot: {plot_file_name}")

    def visualize_iter_loss(self, iter, likelihood, population_size):
        fig, axs = plt.subplots(1,2,figsize=(8,4))
        axs[0].set_ylabel('log10 of total likelihood')
        axs[0].set_xlabel('Iterations')
        axs[0].plot(iter, np.log10(likelihood))
        plt.grid()
        axs[1].set_ylabel('mean likelihood of individuals')
        axs[1].set_xlabel('Iterations')
        axs[1].plot(iter[40::], (likelihood[40::]/population_size))
        plt.grid()
        plt.show()
        #Spennende å se iter vs likelihood/total population

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
        return 0
        

    def data_file(self, individuals, likelihood, pop_size, output_dir=str(Path.cwd())):
        path = str(Path(output_dir) / "data.csv") 
        new_list = []
        for i in range(pop_size):
            str1 = []
            for j in range(self.dim):
                str_ = round(individuals[i,j],10)
                str1.append(str_)
            str2 = likelihood[i]
            new_list.append([str1[0:(self.dim)], round(str2,10)])
        with open(path, 'w', newline='') as csvfile:
            csvfile = csv.writer(csvfile, delimiter=',')
            csvfile.writerow(['Individual' + '       '+'Score'])

            csvfile.writerows(new_list)

