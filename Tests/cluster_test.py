import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append(r'C:\Users\Lenovo\Documents\Master')
from problem_func import *


def Himmelblau(x):
    func = 0
    dim = len(x)
    for i in range(dim-1):
        func += (x[i]**2 + x[i+1] - 11)**2 + (x[i] + x[i+1]**2 -7)**2
    func += 1
    func = np.log(func)

    if len(x) == 2:
        pass
    elif len(x) == 3:
        func -= 0.265331837897597
    elif len(x) == 4:
        func -= 1.7010318616354436
    elif len(x) == 5:
        func -= 2.3001107745553155
    elif len(x) == 6:
        func -= 2.8576426513378994
    return func
class Cluster():
    def __init__(self, individual, likelihood):
        self.individual = individual
        self.likelihood = likelihood
        self.dim  = len(self.individual[0])
        self.num_ind = len(likelihood)
        self.prob_func = 'Himmelblau'
        self.Data = Problem_Function(self.dim)

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
        self.un_empty_cluster = k
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
        
        
        for arg in range(k):
            _,true_lik  = self.eval_likelihood_ind(self.super_centroids[arg,:])
            if true_lik > 4:

                exit()
                temp_lik = self.likelihood[mess]
                new_lik = 'nan'

                for b in range(len(temp_lik)):
                    if temp_lik[b] < true_lik:
                        new_lik = temp_lik[b]
                        self.super_centroids[arg, :] = new_lik
                #No individuals in cluster with sufficiently low likelihood
                if new_lik == 'nan':
                    temp_ind = self.individual[mess]
                    for j in range(self.dim):
                        for b in range(temp_lik):
                            dist_cluster = 100
                            for argy in range(k):
                                if arg != argy: 
                                    #Oppdater tilhørighet gitt nærmeste cluster
                                    dist_comp = np.linalg.norm(temp_ind[b, j]- self.super_centroids[argy, j])
                                    if dist_comp < dist_cluster:
                                        self.super_labels[mess,j] = argy
                                        dist_cluster = dist_comp        

                        #Regn ut ny cluster (med disse nye labels inkludert)
                        #Lag ny cluster array
                    self.un_empty_cluster -= 1
        return self.super_centroids, self.super_labels

    def cluster(self, loglike_tol, k):
        """_summary_

        Args:
            k (_int_): number of clusters 
            loglike_tol (_float_): threshold of the log-likelihood
            
        Returns:
            super_centroids (_array_): The center of mass of each centroid 
            super_labels (_array_): The label of each individual, corresponding to a centroid
        """

        self.k = k
        self.un_empty_cluster = k
        cluster_sort = np.where(self.likelihood< loglike_tol)
        self.X = self.individual[cluster_sort]

        labels    = np.zeros(k)
        self.super_centroids = np.zeros((k,dim))
        self.super_labels = np.zeros((self.num_ind))
        maxiter = 1000
        centroids = self.X[np.random.choice(range(self.num_ind), size=k, replace=False)]
        for _ in range(maxiter):
            labels = np.zeros((self.num_ind))
            distance = np.zeros((self.num_ind, k))
            
            #Finn avstand til alle punkter
            for w in range(k):
                for i in range(self.num_ind):
                    distance[i,w] = np.linalg.norm(self.individual[i,:] - centroids[w])
                
            #Har avstand til alle punkter
            
            #Skriv disse 
            for i in range(self.num_ind):
                max_dist = 1000
                for w in range(k):
                    if distance[i,w] < max_dist:
                        labels[i] = int(w)
                        max_dist = distance[i,w]
            new_centroids = np.zeros_like(centroids)
            for w in range(k):                
                new_dex = np.where(labels[:] == w)
                new_ind = self.individual[new_dex]
                new_centroids[w] = np.mean(new_ind[:])
            if np.linalg.norm(new_centroids == centroids) < 1e-3:
                self.super_centroids = centroids
                self.super_labels[:] = labels[:]
                break
            centroids = new_centroids
        self.super_centroids = centroids
        self.super_labels = labels
        for arg in range(k):
            _,true_lik  = self.eval_likelihood_ind(self.super_centroids[arg,:])
            if true_lik > 4:
                print('bad cluster')
                mess = np.where(self.super_labels == arg)
                temp_lik = self.likelihood[mess]
                new_lik = 'nan'

                for b in range(len(temp_lik)):
                    if temp_lik[b] < true_lik:
                        print('success')
                        new_lik = temp_lik[b]
                        true_lik = new_lik
                        self.super_centroids[arg, :] = new_lik
                        
                #No individuals in cluster with sufficiently low likelihood
                if new_lik == 'nan':
                    print('still bad')
                    temp_ind = self.individual[mess]
                    for b in range(len(temp_lik)):
                        dist_cluster = 100
                        for argy in range(k):
                            if arg != argy: 
                                #Oppdater tilhørighet gitt nærmeste cluster
                                dist_comp = np.linalg.norm(temp_ind[b]- self.super_centroids[argy])
                                if dist_comp < dist_cluster:
                                    self.super_labels[mess] = argy
                                    dist_cluster = dist_comp        

                    #Regn ut ny cluster (med disse nye labels inkludert)
                    #Lag ny cluster array
                    self.un_empty_cluster -= 1
        
"""
    def cluster(self, loglike_tol, k):
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
        err = False
        cluster_sort = np.where(self.likelihood< loglike_tol)
        self.X = self.individual[cluster_sort]
        tolerance = 1e-4
        max_iter = 1000
        # Randomly initialize centroids
        centroids = self.X[np.random.choice(range(len(self.X)), k, replace=False)]



        # best_i, best_j, best_k, best_w = 0,1,2,3
        # max_dist = 0
        # #Finn de to punkter som er lengst fra hverandre
        # for i in range(len(self.X)):
        #     for j in range(len(self.X)):
        #         dist = np.linalg.norm(self.X[i] -self.X[j])
        #         if dist > max_dist:
        #             max_dist = dist
        #             best_i = i
        #             best_j = j
        # max_dist = 0
        # #Finn de fire punktene som er lengst fra disse igjen
        # for i in range(len(self.X)):
        #     for j in range(len(self.X)):
        #         dist_i1 = np.linalg.norm(self.X[best_i]- self.X[i])
        #         dist_i2 = np.linalg.norm(self.X[best_i]- self.X[j])
        #         dist_j1 = np.linalg.norm(self.X[best_j]- self.X[i])
        #         dist_j2 = np.linalg.norm(self.X[best_j]- self.X[j])
        #         dist_ = np.linalg.norm(self.X[i] - self.X[j])
                
        #         dist = dist_i1 + dist_i2 + dist_j1 + dist_j2 + dist_
        #         if dist > max_dist:
        #             max_dist = dist
        #             best_k = i
        #             best_w = j
        # cent_list = [best_i, best_j, best_k, best_w]
        # centroids = self.X[cent_list]



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
        
        for arg in range(k):
            _,true_lik  = self.eval_likelihood_ind(centroids[arg])
            if true_lik > 1.15:
                print('Bad cluster')
                print('Likelihood value of cluster:', true_lik)
                print('In cluster:', arg)
                err = True
                break
        exit()

        return centroids, clusters

"""

    def eval_likelihood_ind(self, individual):
        Data = self.Data
        return Data.evaluate(individual, self.prob_func)

np.random.seed(11)
xmin = -5
xmax = 5
dim  = 2
pop  = 50
individual = np.zeros((pop, dim))
likelihood = np.zeros(pop)

for i in range(pop):
    for j in range(dim):
        individual[i,j] = np.random.uniform(-5,5)
    value = Himmelblau(individual[i])

    likelihood[i] = value


test = Cluster(individual, likelihood)
test.cluster(loglike_tol=8, k =5)
