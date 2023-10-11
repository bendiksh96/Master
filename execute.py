from time import perf_counter
from devo import *
from problem_func import *
from vis_writ import *
import numpy as np
import matplotlib.pyplot as plt

#¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
#================================================================
#Standardiserte parametere for denne oppgaven er 
#Dimensjoner: 4, 7, 9, 11
#Foreldre: 20000
#Iterasjoner: Avhenger av Dimensjoner. TBD
#Problemer: Rosenbrock og Himmelblau
#
#Metoder:
#DE                         - Gjort
#jDE                        - Gjort
#jDErpo                     - Gjort
#dynNP-jDE (?)
#JADE                       - Dårligere versjon av SHADE, dermed mulig å droppe!
#SHADE                      - Gjort 
#SHADE-ILS
#DDMS                       - Spennende
#BDE                        - Brukt til minimum 1000D i artikkel - Anders liker
#BLOP                       - Spennende shit - Anders liker
#   -SABLA
#================================================================
#¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤


start = perf_counter()
max_iter = 100
x_min = -5
x_max = 10
dim =   9
pop =   [1000, 100]
n_p =   [0.4 * p for p in pop]
m_p =   [0.1]
no_pop = 1
np.random.seed(4452)


#Use keywords: "DE", "jDE", "jDErpo", "SHADE"
eval_type = "SHADE"

#End program if global likelihood does not improve by a factor of 
end_crit = 1e-4

iter_var = np.linspace(0,max_iter, max_iter)

#Må fikse for multidim populasjoner
sum_likelihood = []
evolution_individuals = []

for i in range(no_pop):
    evolution_individuals.append(np.zeros((max_iter, pop[i], dim)))
    sum_likelihood.append(np.zeros((max_iter, pop[i])))

#Evolution
klam = Data(dim, x_min, x_max)
vary = DEVO(maxiteration=max_iter, population_size_list=pop, mut_prob=m_p, no_parents=n_p, dim = dim, problem_func = "Himmelblau", no_pop=no_pop)
vary.initialize_single_pop(x_min, x_max, True)
vary.eval_likelihood_pop()
print(np.std(vary.likely_ind[0]))

for iter in range(max_iter):
    vary.fornicate(method= eval_type)
    vary.eval_likelihood_pop()
    vary.check_oob()    
    # vary.migration(0.1)
    vary.iter += 1
    for k in range(no_pop):
        evolution_individuals[k][iter] = vary.ind[k]
        sum_likelihood[k][iter] = sum(vary.likely_ind[k])
        
    #Litt sugent end_crit
    k= iter
    for k in range(no_pop):    
        if iter > int(iter+10):
            
            if (np.mean(sum_likelihood[k][(iter-10):(iter-1)])-sum_likelihood[k][iter]) < end_crit:
                print(f'Ended evolution after {iter} iterations, due to population {k}')
                break
end = perf_counter()

print(f'Time of evaluation: {end-start:.2f} seconds')


#Visualization
# klam.visualize_population_evolution(evolution_individuals[0], vary.likely_ind[0])
# klam.visualize_1(vary.ind[0],vary.likely_ind[0], eval_type=eval_type)
# klam.visualize_iter_loss(iter_var[0:k], sum_likelihood[0:k], n_p)
# klam.data_file(vary.ind[0], vary.likely_ind[0], vary.population_size_list[0])
# klam.visualize_2(max_iter, vary.ind[0], vary.likely_ind[0])
