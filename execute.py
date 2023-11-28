from pathlib import Path
import sys
import numpy as np
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


# np.random.seed(154321431)
# method = 'jde'
method = 'shade'
# method = 'double_shade'

optimizer = DEVO_class(dim = 3, problem_func = 'Himmelblau', method = method)
optimizer.intialize_population(xmin = -5, xmax = 5, num_ind = 100)
optimizer.evolve(maxiter=1000, tol=1e-8)

n_points = len(optimizer.hist_lik)
print(f"Number of points: {n_points}")

hist_ind_arr = np.array(optimizer.hist_ind)
hist_lik_arr = np.array(optimizer.hist_lik)

vis = Data(optimizer.dim, optimizer.xmin, optimizer.xmax)
vis.visualize_1(hist_ind_arr, hist_lik_arr, method, output_dir=Path.cwd())
vis.data_file(hist_ind_arr, hist_lik_arr, n_points) 


#¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
# Fiks et konvergeringskriterie - Funker sånn passe?
# 
#
#¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤