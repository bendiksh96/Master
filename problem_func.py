import numpy as np

class Problem_Function:
    def __init__(self, dim):
        self.dim =  dim
    
    def Rosenbrock(self, x):
        val = 0
        for i in range(self.dim-1):
            val += 100*(x[i+1]-x[i]**2)**2 + (1 - x[i])**2
        return val
        
    def Eggholder(self, x, dim=2):
        func = 0
        for i in range(self.dim-1):
            func -= (x[i+1]+47)*np.sin(np.sqrt(np.abs(x[i+1]+x[i]/2+47)))+ x[i]*np.sin(np.sqrt(np.abs(x[i]-(x[i+1]+47))))
        return func
