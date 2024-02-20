import numpy as np
import sys
sys.path.append(r'C:\Users\Lenovo\Documents\Master')
from problem_func import *



class d_SHADE_pso:
    def __init__(self, individual, likelihood, problem_func):
        self.dim        = len(individual[0])
        self.individual = individual
        self.likelihood = likelihood
        self.true_likelihood = likelihood
        self.num_ind    = len(likelihood)
        self.prob_func  = problem_func
        self.k_arg      = 0 
        self.velocity   = np.zeros_like(self.individual)
        self.optimum    = 10


        self.optimal_individual    = np.zeros_like(self.individual)
        self.force    = np.zeros_like(self.individual)
        
        self.A          = []
        self.Flist      = [0.1 for p in range(self.num_ind)]
        self.CRlist     = [0.1 for p in range(self.num_ind)]
        self.M_CR       = [0.1 for p in range(self.num_ind)]
        self.M_F        = [0.1 for p in range(self.num_ind)]
        
        self.Data = Problem_Function(self.dim)


    def evolve_converge(self):
        S_CR       = []
        S_F        = []
        delta_f    = []

        self.u          = np.zeros_like(self.individual)
        self.v          = np.zeros_like(self.individual)
        sort            = np.argsort(self.likelihood, axis = 0)
        best_indexes    = sort[0:self.num_ind]
        xpbest          = self.individual[best_indexes]
        self.abs_best   = self.likelihood[best_indexes[0]]
        self.best_ind   = self.individual[best_indexes][0]
        #Mutant vector
        for i in range(self.num_ind-1):
            ri = np.random.randint(1,self.num_ind) 
            self.CRlist[i] = np.random.normal(self.M_CR[ri], 0.1)
            #Burde være Cauchy-fordeling
            self.Flist[i]  = np.random.normal(self.M_F[ri], 0.1)
            
            #Current to pbest/1 
            ri1 = np.random.randint(self.num_ind)
            ri2 = np.random.randint(self.num_ind)
            ri3 = np.random.randint(self.num_ind)
            
            self.v[i] = self.individual[i] + self.Flist[i]*(xpbest[ri3]-self.individual[i]) + self.Flist[i]*(self.individual[ri1]- self.individual[ri2])
                                            
        #Crossover
        for i in range(self.num_ind):
            for j in range(self.dim):
                randint = np.random.randint(0,1)
                if randint < self.CRlist [i]:
                    # self.u[i,j] = self.v[i,j]
                
                    perceived_likelihood, true_likelihood  = self.eval_likelihood_ind(self.v[i])     
                    if perceived_likelihood <= self.likelihood[i]:
                        self.individual[i] = self.v[i]
                        self.A.append(self.individual[i])        
                        delta_f.append(perceived_likelihood-self.likelihood[i])                
                        S_CR.append(self.CRlist[i])
                        S_F.append(self.Flist[i])
                        self.likelihood[i] = perceived_likelihood
                        if len(self.A) > self.num_ind :
                            del self.A[np.random.randint(0, self.num_ind)]
        #Update weights
        if len(S_CR) != 0:
            if self.k_arg>=self.num_ind:
                self.k_arg = 1
            wk = []
            mcr = 0
            mf_nom = 0
            mf_denom = 0
            tol = 1e-3
            for arg in range(len(S_CR)):
                wk.append(delta_f[arg]/(sum(delta_f)+tol))
            for arg in range(len(S_CR)):
                mcr += wk[arg] * S_CR[arg]
                mf_nom  += wk[arg]*S_F[arg]**2
                mf_denom += wk[arg]*S_F[arg]
            self.M_CR[self.k_arg] = mcr
            self.M_F[self.k_arg] = mf_nom/(mf_denom+tol)
            self.k_arg += 1

    def evolve_explore(self):
        #Initialize the list of weights
        S_CR       = []
        S_F        = []
        delta_f    = []

        #Initialize temporary variables
        self.u          = np.zeros_like(self.individual)
        self.v          = np.zeros_like(self.individual)
        
        #Previous best indices
        sort            = np.argsort(self.likelihood, axis = 0)
        best_indexes    = sort[0:self.num_ind]
        xpbest          = self.individual[best_indexes]
        self.abs_best   = self.likelihood[best_indexes[0]]
        self.best_ind   = self.individual[best_indexes][0]

        #Evolve individuals
        for i in range(self.num_ind-1):
            
            ri = np.random.randint(1,self.num_ind) 
            self.CRlist[i] = np.random.normal(self.M_CR[ri], 0.1)
            #Burde være Cauchy-fordeling
            self.Flist[i]  = np.random.normal(self.M_F[ri], 0.1)
            
            #Current to pbest/1 
            ri1 = np.random.randint(self.num_ind)
            ri2 = np.random.randint(self.num_ind)
            ri3 = np.random.randint(self.num_ind)
            self.v[i] = self.individual[i] + self.Flist[i]*(xpbest[ri3]-self.individual[i]) + self.Flist[i]*(self.individual[ri1]- self.individual[ri2])
                                            
        #Crossover
        for i in range(self.num_ind):
            for j in range(self.dim):
                randint = np.random.randint(0,1)
                if randint < self.CRlist [i]:
                    # self.u[i,j] = self.v[i,j]
            
                    #Check if the new crossover individual is superior to the prior
                    perceived_likelihood, true_likelihood  = self.eval_likelihood_ind(self.v[i])     
                    if perceived_likelihood <= self.likelihood[i]:
                        self.individual[i] = self.v[i]
                        self.A.append(self.individual[i])        
                        delta_f.append(perceived_likelihood-self.likelihood[i])                
                        S_CR.append(self.CRlist[i])
                        S_F.append(self.Flist[i])
                        self.likelihood[i] = perceived_likelihood
                        self.true_likelihood[i] = true_likelihood

                        #If archive exceeds number of individuals, delete a random archived log.
                        if len(self.A) > self.num_ind :
                            del self.A[np.random.randint(0, self.num_ind)]
        
        #Update weights
        if len(S_CR) != 0:
            if self.k_arg>=self.num_ind:
                self.k_arg = 1
            wk = []
            mcr = 0
            mf_nom = 0
            mf_denom = 0
            tol = 1e-3
            for arg in range(len(S_CR)):
                wk.append(delta_f[arg]/(sum(delta_f)+tol))
            for arg in range(len(S_CR)):
                mcr += wk[arg] * S_CR[arg]
                mf_nom  += wk[arg]*S_F[arg]**2
                mf_denom += wk[arg]*S_F[arg]
            self.M_CR[self.k_arg] = mcr
            self.M_F[self.k_arg] = mf_nom/(mf_denom+tol)
            self.k_arg += 1


    def cluster_dim(self, loglike_tol, k):
        """_summary_

        Args:
            k (_int_): number of clusters 
            loglike_tol (_float_): threshold of the log-likelihood
            
        Returns:
            super_centroids (_array_): The center of mass of each centroid 
            super_labels (_array_): The label of each individual, corresponding to a centroid
        """
        self.k = k
        cluster_sort = np.where(self.likelihood< loglike_tol)
        self.X = self.individual[cluster_sort]

        labels    = np.zeros(k)
        self.super_centroids = np.zeros((k, self.dim))
        self.super_labels = np.zeros((self.num_ind, self.dim))
        maxiter = 1000
        for j in range(self.dim):
            for _ in range(maxiter):
                labels = np.zeros((self.num_ind, self.dim))
                centroids = self.individual[np.random.choice(range(self.num_ind), size=k, replace=False),j]
                distance = np.zeros((self.num_ind, k))
                
                #Finn avstand til alle punkter
                for w in range(k):
                    for i in range(self.num_ind):
                        distance[i,w] = np.linalg.norm(self.individual[i,j] - centroids[w])
                    
                #Har avstand til alle punkter
                
                #Skriv disse 
                for i in range(self.num_ind):
                    max_dist = 1000
                    for w in range(k):
                        if distance[i,w] < max_dist:
                            labels[i,j] = int(w)
                            max_dist = distance[i,w]
                new_centroids = np.zeros_like(centroids)
                for w in range(k):                
                    new_dex = np.where(labels[:,j] == w)
                    new_ind = self.individual[new_dex]
                    new_centroids[w] = np.mean(new_ind[:,j])
                if all(new_centroids == centroids):
                    self.super_centroids[:,j] = centroids
                    self.super_labels[:,j] = labels[:,j]
                    break
            self.super_centroids[:,j] = centroids
            self.super_labels[:,j] = labels[:,j]
        return self.super_centroids, self.super_labels

    def cluster(self, loglike_tol):
        #def kmeans(X, k,    max_iterations=100, tolerance=1e-4):
        """
        K-means clustering algorithm.

        Parameters:
        - X: numpy array, input data points
        - k: int, number of clusters
        - max_iterations: int, maximum number of iterations
        - tolerance: float, convergence tolerance

        Returns:
        - centroids: numpy array, final centroids of clusters
        - clusters: list of lists, indices of data points in each cluster
        """
        cluster_sort = np.where(self.likelihood< loglike_tol)
        self.X = self.individual[cluster_sort]
        tolerance = 1e-4
        max_iter = 1000
        self.k = 4
        k = self.k
        best_i, best_j, best_k, best_w = 0,1,2,3
        # Randomly initialize centroids
        # centroids = self.X[np.random.choice(range(len(self.X)), k, replace=False)]
        max_dist = 0
        #Finn de to punkter som er lengst fra hverandre
        for i in range(len(self.X)):
            for j in range(len(self.X)):
                dist = np.linalg.norm(self.X[i] -self.X[j])
                if dist > max_dist:
                    max_dist = dist
                    best_i = i
                    best_j = j
        max_dist = 0
        #Finn de fire punktene som er lengst fra disse igjen
        for i in range(len(self.X)):
            for j in range(len(self.X)):
                dist_i1 = np.linalg.norm(self.X[best_i]- self.X[i])
                dist_i2 = np.linalg.norm(self.X[best_i]- self.X[j])
                dist_j1 = np.linalg.norm(self.X[best_j]- self.X[i])
                dist_j2 = np.linalg.norm(self.X[best_j]- self.X[j])
                dist_ = np.linalg.norm(self.X[i] - self.X[j])
                
                dist = dist_i1 + dist_i2 + dist_j1 + dist_j2 + dist_
                if dist > max_dist:
                    max_dist = dist
                    best_k = i
                    best_w = j
        cent_list = [best_i, best_j, best_k, best_w]
        centroids = self.X[cent_list]
        for _ in range(max_iter):
            # Assign each data point to the nearest centroid
            distances = np.linalg.norm(self.X[:, np.newaxis] - centroids, axis=2)
            
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            
            
            new_centroids = np.array([self.X[labels == i].mean(axis=0) for i in range(k)])
            # Check for convergence
            if np.linalg.norm(new_centroids - centroids) < tolerance:
                break

            centroids = new_centroids
        # Assign data points to clusters for the final centroids
        final_distances = np.linalg.norm(self.X[:, np.newaxis] - centroids, axis=2)
        final_labels = np.argmin(final_distances, axis=1)
        
        # Create a list of indices for each cluster
        clusters = [np.where(final_labels == i)[0] for i in range(k)]
        return centroids, clusters
        

    def init_particle(self):
        #Velocity initialization
        for i in range(self.num_ind):            
            for j in range(self.dim):
                ru = np.random.uniform(int(-1e-3),int(1e-3))
                ru = np.random.uniform(0,1)
                self.velocity[i,j] = (self.individual[i,j] - self.optimal_individual[i,j]) * ru 
        print('Particles Initialized')
    
    def evolve_particle(self):
        
        #Gjør hver posisjon om til en partikkel
        
        #Finn beste cluster for partikkelet ved å gå gjennom avstand til hvert massesentrum
        
        #Gi hver partikkel en hastighet, skalert til sitt massesentrum
        
        for i in range(self.num_ind):
            k = 0.1
            for j in range(self.dim):
                self.force[i,j]         = -k * (self.likelihood[i] - self.optimum) * (self.individual[i,j] - self.optimal_individual[i,j])#/abs(self.individual[i,j] - self.optimal_individual[i,j])
                self.velocity[i,j]      = self.velocity[i,j] + self.force[i,j]
                self.individual[i,j]    = self.individual[i,j] + self.velocity[i,j]
            self.likelihood[i], self.true_likelihood[i] = self.eval_likelihood_ind(self.individual[i])

        
        # Modeller kraft som f = -k (log_likelihood - log_likelihood_best)* unit_vector
            #Fortegnet avhenger om gauss er skrudd på eller ei
        #Bruk Euler-Cromer
        
      
    #Metode for å evaluere likelihood til et enkelt individ.
    #Liker ikke helt å måtte kalle på den her også, men det er hittil det beste jeg har.
    def eval_likelihood_ind(self, individual):
        Data = self.Data
        return Data.evaluate(individual, self.prob_func)
        
