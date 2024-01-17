import numpy as np

class Problem_Function:
    def __init__(self, dim):
        self.dim =  dim
    
    def param_change(self, best,delta_log):
        self.best   = best
        self.sigma  = delta_log/3
        self.delta_log = delta_log
        
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
    
    def mod_eggholder(self, x, best, sigma):
        func = 0
        sigma = 0.3
        for i in range(self.dim-1):
            func -= (x[i+1]+47)*np.sin(np.sqrt(abs(x[i+1]+(x[i]/2)+47)))+ x[i]*np.sin(np.sqrt(abs(x[i]-(x[i+1]+47))))    
        true_func = func
        
        #If the value supercedes the necessity of a gaussian, add one.  
        if true_func >= self.delta_log:
            func = func - func * np.exp(-(best-(func))**2/(2*sigma**2)) 
        return func, true_func
    
    def Himmelblau(self, x):
        func = 0
        for i in range(self.dim-1):
            func += (x[i]**2 + x[i+1] - 11)**2 + (x[i] + x[i+1]**2 -7)**2
        func += 1
        func = np.log(func)
        return func, func
    
    def mod_himmelblau(self, x):
        func = 0
        delta_log = self.delta_log
        for i in range(self.dim-1):
            func += (x[i]**2 + x[i+1] - 11)**2 + (x[i] + x[i+1]**2 -7)**2 
        func += 1
        func = np.log(func)
        true_func = func
        
        
        if true_func <= (self.best+delta_log):
            
            func = self.best + delta_log
            func = func  + delta_log * np.exp(- (self.best-(true_func))**2/(2*self.sigma**2)) 
    
    
        return func , true_func
        
        
    