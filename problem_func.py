import numpy as np

class Problem_Function:
    def __init__(self, dim):
        self.dim =  dim
    
    def param_change(self, best, delta, sigma):
        self.best   = best
        self.delta  = delta
        self.sigma  = sigma
    def evaluate(self, x, problem_func):
        if problem_func == 'Eggholder':
            val = self.Eggholder(x)
        if problem_func == 'mod_Eggholder':
            val = self.mod_eggholder(x)
        if problem_func == 'Rosenbrock':
            val = self.Rosenbrock(x)
        if problem_func == 'mod_Rosenbrock':
            val = self.mod_rosenbrock(x)
        if problem_func == 'Himmelblau':
            val = self.Himmelblau(x)
        if problem_func == 'mod_Himmelblau':
            val = self.mod_himmelblau(x)
        return val
    def Rosenbrock(self, x):
        func = 0
        for i in range(self.dim-1):
            func += 100*(x[i+1]-x[i]**2)**2 + (1 - x[i])**2
        
        return func
    
    def mod_rosenbrock(self, x):
        func = 0
        for i in range(self.dim-1):
            func += 100*(x[i+1]-x[i]**2)**2 + (1 - x[i])**2
        
        true_func = func
        func = func - func * np.exp(-(self.best-(func))**2/(2*self.sigma**2)) 
        return func, true_func
          
    def Eggholder(self, x):
        func = 0
        for i in range(self.dim-1):
            func -= (x[i+1]+47)*np.sin(np.sqrt(abs(x[i+1]+(x[i]/2)+47)))+ x[i]*np.sin(np.sqrt(abs(x[i]-(x[i+1]+47))))
        
        return func
    
    def mod_eggholder(self, x, best, delta, sigma):
        func = 0
        sigma = 0.5
        for i in range(self.dim-1):
            func -= (x[i+1]+47)*np.sin(np.sqrt(abs(x[i+1]+(x[i]/2)+47)))+ x[i]*np.sin(np.sqrt(abs(x[i]-(x[i+1]+47))))    
        true_func = func
        func = func - func * np.exp(-(best-(func))**2/(2*sigma**2)) 
        return func, true_func
    
    def Himmelblau(self, x):
        func = 0
        for i in range(self.dim-1):
            func += (x[i]**2 + x[i+1] - 11)**2 + (x[i] + x[i+1]**2 -7)**2
        func += 1
        func = np.log(func)
        return func
    
    def mod_himmelblau(self, x):
        func = 0
        for i in range(self.dim-1):
            func += (x[i]**2 + x[i+1] - 11)**2 + (x[i] + x[i+1]**2 -7)**2 
        func += 1
        func = np.log(func)

        true_func = func
        
        kunk = func *(1 + np.exp(-(self.best-(func+(1e-3)))**2/(2*self.sigma**2)) )
        # print(np.exp(-(best-(func))**2/(2*sigma**2)), kunk)
        return kunk, true_func
    