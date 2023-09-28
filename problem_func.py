import numpy as np

class Problem_Function:
    def __init__(self, dim):
        self.dim =  dim
    
    def Rosenbrock(self, x):
        func = 0
        for i in range(self.dim-1):
            func += 100*(x[i+1]-x[i]**2)**2 + (1 - x[i])**2
        func_perc = func
        if func < 10:
            func_perc = 0.01
        return func, func_perc 
    
          
    def Eggholder(self, x):
        func = 0
        for i in range(self.dim-1):
            func -= (x[i+1]+47)*np.sin(np.sqrt(abs(x[i+1]+(x[i]/2)+47)))+ x[i]*np.sin(np.sqrt(abs(x[i]-(x[i+1]+47))))
        func_perc = func
        if func < 0.5:
            func_perc = 0.01
        return func, func_perc
    
    
    def Himmelblau(self, x):
        func = 0
        for i in range(self.dim-1):
            func += (x[i]**2 + x[i+1] - 11)**2 + (x[i] + x[i+1]**2 -7)**2
        
        func_perc = func
        if func < 1:
            func_perc = 0
        return func, func_perc