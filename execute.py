from time import perf_counter
from devo import *
from problem_func import *
from vis_writ import *
import numpy as np
import matplotlib.pyplot as plt

start = perf_counter()
max_iter = 40
x_min = -5
x_max = 10
dim =   5
pop =   10000
n_p =   int(0.4*pop)
m_p =   0.2
np.random.seed(4456)

#End program if global likelihood does not improve by a factor of 
end_crit = 1e-2

iter_var = np.linspace(0,max_iter, max_iter)
sum_likelihood = np.linspace(0,max_iter, max_iter)
evolution_individuals = np.zeros((max_iter, pop, dim))


#Evolution
klam = Data(dim, x_min, x_max)
vary = DEVO(population_size=pop, mut_prob=m_p, no_parents=n_p, dim = dim, problem_func = "Rosenbrock", no_pop=1)
vary.initialize_single_pop(dim, x_min, x_max)
vary.eval_likelihood_pop()

for iter in range(max_iter):
    vary.fornicate(method= 'jDE')
    vary.eval_likelihood_pop()
    vary.check_oob()
    evolution_individuals[iter] = vary.ind
    sum_likelihood[iter] = sum(vary.likely_ind)
    #Litt sugent end_crit
    k= iter
    if iter > int(iter+10):
        if (np.mean(sum_likelihood[(iter-10):(iter-1)])-sum_likelihood[iter]) < end_crit:
            print(f'Ended evolution after {iter} iterations')
            break
end = perf_counter()

print(f'Time of evaluation: {end-start:.2f} seconds')

klam.visualize_population_evolution(evolution_individuals, vary.likely_ind)
#Visualization
# klam.visualize_1(vary.ind,vary.likely_ind)
# klam.visualize_iter_loss(iter_var[0:k], sum_likelihood[0:k], n_p)
# klam.data_file(vary.ind, vary.likely_ind, vary.population_size)
# klam.visualize_2(max_iter, vary.ind, vary.likely_ind)
