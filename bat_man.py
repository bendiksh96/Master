import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

np.random.seed(12313)

def visualize_1(individuals, likelihood, xmin,xmax, dim, eval_type='Eggholder'):
    sort_ = np.argsort((-1*likelihood), axis=0)
    likelihood = likelihood[sort_]
    individuals = individuals[sort_,:].reshape((len(likelihood), dim))
    # kn =  np.where(likelihood > 60)
    # individuals[kn,:] = 'nan'
            
    # Create figure
    fontsize = 8
    fsize_per_dim = 5.0
    fsize = fsize_per_dim * (dim-1)
    markersize = 10.0 / (dim-1)
    markerborderwidth = 0.25 / (dim-1)
    markerbordercolor = '0.3'
    figpad = 0.9 / fsize  # 0.07
    plotpad = 0.9 / fsize # 0.07
    plot_width = ( 1.0 - 2*figpad - max(0,dim-2)*plotpad ) / (dim - 1.)
    plot_height = plot_width

    fig = plt.figure(figsize=(fsize, fsize))
    cmap_vmin = 0.0
    cmap_vmax = 6
    plot_facecolor = '0.0'
    for i in range(dim):
        for j in range(i+1,dim):
        # Add axes
            cmap = plt.get_cmap("viridis_r")
            #cmap.set_extremes(under= 'black', over='grey')
            norm = plt.Normalize(-2000,200)
            # norm = matplotlib.cm.colors.Normalize(vmin=cmap_vmin, vmax=cmap_vmax)
            left = figpad + i*(plot_width + plotpad)
            bottom = figpad + (j-1)*(plot_width + plotpad)
            ax = fig.add_axes((left, bottom, plot_width, plot_height))
            ax.set_facecolor(plot_facecolor)

            # Axis ranges
            plt.xlim([1.05*xmin, 1.05*xmax])
            plt.ylim([1.05*xmin, 1.05*xmax])

            # Axis labels
            plt.ylabel("x_" + str(j), fontsize=fontsize)
            plt.xlabel("x_" + str(i), fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)

            # Create a colour scale normalization
            # norm = matplotlib.cm.colors.Normalize(vmin=cmap_vmin, vmax=cmap_vmax)
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
    # output_dir_name = 'asad.png'
    plt.show()



def Eggholder(x_):
    func = 0
    for i in range(dim-1):
        func -= (x_[i+1]+47)*np.sin(np.sqrt(abs(x_[i+1]+(x_[i]/2)+47)))+ x_[i]*np.sin(np.sqrt(abs(x_[i]-(x_[i+1]+47))))
    
    return func

def Rosenbrock(x_):
        func = 0
        for i in range(dim-1):
            func += 100*(x_[i+1]-x_[i]**2)**2 + (1 - x_[i])**2
        return func
    
num_pop     = 4
pop_size    = 100
NFE_max     = 1e3
dim         = 5
NFE         = 0 
BM          = [] 
BM_val      = []
individual  = []
likelihood  = []
velocity    = []
pulse       = []
xmin,xmax   = -5,5
gen         = 0 
rj          = .2
rj0         = .2
A           = [.5 for  p in range(num_pop)]
eps         = .01
gamma       = .1
alfa        = .5

#A - loudness
#Needs to be tuned in order to 


for pop in range(num_pop):
    role_pop = []
    kar = np.zeros((pop_size,dim))
    vel = np.zeros((pop_size,dim))
    pul = np.zeros((pop_size,dim))
    verdi = np.zeros((pop_size,1))
    for i in range(pop_size):
        for j in range(dim):
            kar[i,j] = np.random.uniform(xmin, xmax)
            vel[i,j] = np.random.uniform(-1,1)
            pul[i,j] = rj
        verdi[i,0] = Eggholder(kar[i,:])        
    individual.append(kar)
    likelihood.append(verdi)
    velocity.append(vel)
    pulse.append(pul)
    BM.append(kar)
    BM_val.append(verdi)
gen = 1
while NFE < NFE_max:
    for pop in range(num_pop):
        sort = np.argsort(likelihood[pop])
        best_val = likelihood[pop][sort][0]
        best_ind = individual[pop][sort][0]
        worst_val = likelihood[pop][sort][-1]
        for i in range(pop_size):
            for j in range(dim):
                u = np.random.uniform(0,1)
                #Frequency
                f_j = best_val + (best_val - worst_val) * np.random.uniform(0,1)
                #Velocity
                velocity[pop][i,j] = velocity[pop][i,j] + 1/(100*pop+pop_size)*(individual[pop][i,j] - best_ind[0][j]) * f_j[0][0]
                
                if u < pulse[pop][i,j]:
                    individual[pop][i,j] = best_ind[0][j] + eps*A[pop]
                else:
                    individual[pop][i,j] = velocity[pop][i,j] + individual[pop][i,j] 
                if individual[pop][i,j] < xmin:
                    velocity[pop][i,j] = - velocity[pop][i,j]
                    individual[pop][i,j] = xmin+  abs(individual[pop][i,j] - xmin) + xmin
                if individual[pop][i,j] > xmax:
                    velocity[pop][i,j] = - velocity[pop][i,j]

                    individual[pop][i,j] = xmax -  abs(individual[pop][i,j] - xmax )
                    
            likelihood[pop][i] = Eggholder(individual[pop][i])
            if any(abs(likelihood[pop][i] - BM_val[pop][i]) < [1e-3 for p in range(dim)]):
                velocity[pop][i] = velocity[pop][i] +A[pop]
            if likelihood[pop][i] < BM_val[pop][i] or np.random.uniform(0,1) < A[pop]:
                pulse[pop][i] = rj0*(1- np.exp(-gamma*gen))
                A[pop] = alfa*A[pop]
                BM[pop] = individual[pop]
                BM_val[pop] = likelihood[pop]
           
        gen += 1
        NFE += num_pop * pop_size

individual = BM
likelihood = BM_val
print(BM[1])
#print(BM_val)
plt.scatter(individual[0][:, 0], individual[0][:,1], likelihood[0][:], 'b')
plt.scatter(individual[1][:, 0], individual[1][:,1], likelihood[1][:], 'r')
plt.scatter(individual[2][:, 0], individual[2][:,1], likelihood[2][:], 'g')
plt.scatter(individual[3][:, 0], individual[3][:,1], likelihood[3][:], 'c')

plt.xlabel('x0')
plt.ylabel('x1')
plt.title('')
plt.ylim(xmin, xmax)
plt.xlim(xmin,xmax)
plt.title('Bat Algorithm on Eggholder. Unnomralized.')
plt.show()

#visualize_1(individual[0], likelihood[0],xmin,xmax,dim)