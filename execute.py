from class_devo import *


# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
# Mulige funksjoner:
#   - Himmelblau    [-5,5]
#   - Rosenbrock    [-5,5]
#   - Eggholder     [-512,512]
#
# Mulige evalueringsmetoder:
#   -jde        (Ikke helt oppe og går)
#   -shade
#   -dshade
#   -bat        (Ikke helt oppe og går)
# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤

a = DEVO_class(dim = 5, problem_func = 'Himmelblau', method = 'shade')
a.intialize_population(xmin = -5, xmax = 5, num_ind = 100)
a.evolve(maxiter = 100)

kaare = len(a.hist_lik)
print(kaare)
ind_arr = np.ones((kaare, a.dim))
lik_arr = np.ones((kaare))

for are in range(kaare):   
    ind_arr[are] = a.hist_ind[are]
    lik_arr[are] = a.hist_lik[are]

vis = Data(a.dim, a.xmin, a.xmax)
vis.visualize_1(ind_arr,lik_arr, 'dshade')
vis.data_file(ind_arr, lik_arr, kaare) 
#¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
# Fiks et konvergeringskriterie - Funker sånn passe?
# 
#
#¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤