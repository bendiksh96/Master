import sys
from time import perf_counter
from devo import *
from vis_writ import *
from problem_func import *


default = input("Would you like to use default parameters? [y/n]\n")

if default == 'y':
    start = perf_counter()
    max_iter = 100
    x_min = -5
    x_max = 10
    dim =   9
    pop =   [100, 100]
    n_p =   [0.4 * p for p in pop]
    m_p =   [0.1]
    no_pop = 1
    np.random.seed(4452)

else:
    no_pop = int(input("How many populations?"))




prob_func = input("\nChoose problem function. Your choices are Rosenbrock, Eggholder and Himmelblau. Type which you prefer.\n")
cont_run  = input("\nWould you like the program to run indefinetly? [y/n] \n")
eval_type = input("\nChoose your method. Your choices are DE, jDE, jDERpo and SHADE\n")


if cont_run == 'y':
    cont_run = True
else:
    cont_run  = False


print("Program is running. If you want to stop the run - Press any key. ")

run = True
iter_var = np.linspace(0,max_iter, max_iter)

#Må fikse for multidim populasjoner
sum_likelihood = []
evolution_individuals = []

for i in range(no_pop):
    evolution_individuals.append(np.zeros((max_iter, pop[i], dim)))
    sum_likelihood.append(np.zeros((max_iter, pop[i])))
klam = Data(dim, x_min, x_max)
vary = DEVO(maxiteration=max_iter, population_size_list=pop, mut_prob=m_p, no_parents=n_p, dim = dim, problem_func = "Rosenbrock", no_pop=no_pop)
vary.initialize_single_pop(x_min, x_max, True)
vary.eval_likelihood_pop()

while run == True:
    vary.fornicate(method= eval_type)
    vary.eval_likelihood_pop()
    vary.check_oob()    
    vary.iter += 1
    # for k in range(no_pop):
    #     evolution_individuals[k][iter] = vary.ind[k]
    #     sum_likelihood[k][iter] = sum(vary.likely_ind[k])
        
    #Litt sugent end_crit
    # k= iter
    # for k in range(no_pop):    
    #     if iter > int(iter+10):
            
    #         if (np.mean(sum_likelihood[k][(iter-10):(iter-1)])-sum_likelihood[k][iter]) < end_crit:
    #             print(f'Ended evolution after {iter} iterations, due to population {k}')
    #             break
    if input():
        # klam.visualize_population_evolution(evolution_individuals[0], vary.likely_ind[0])
        # klam.visualize_1(vary.ind[0],vary.likely_ind[0], eval_type=eval_type)
        # klam.visualize_iter_loss(iter_var[0:k], sum_likelihood[0:k], n_p)
        # klam.data_file(vary.ind[0], vary.likely_ind[0], vary.population_size_list[0])
        # klam.visualize_2(max_iter, vary.ind[0], vary.likely_ind[0])

        break


end = perf_counter()


