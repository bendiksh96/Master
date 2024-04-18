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
            val = self.mod_himmelblau2(x)
        if problem_func == 'Rastrigin':
            val = self.Rastrigin(x)
        if problem_func == 'mod_Rastrigin':
            val = self.mod_Rastrigin(x)
        if problem_func == 'Levy':
            val = self.Levy(x)
        if problem_func == 'mod_Levy':
            val = self.mod_Levy(x)
        if problem_func == 'Ackley':
            val = self.Ackley(x)
        if problem_func == 'mod_Ackley':
            val = self.mod_Ackley(x)
        if problem_func == 'Hartman_3D':
            val = self.Hartman_3D(x)
        if problem_func == 'mod_Hartman_3D':
            val = self.mod_Hartman_3D(x)
        if problem_func == 'Michalewicz':
            val = self.Michalewicz(x)
        if problem_func == 'mod_Michalewicz':
            val = self.mod_Michalewicz(x)
        if problem_func == 'Rotated_Hyper_Ellipsoid':
            val = self.Rotated_Hyper_Ellipsoid(x)
        if problem_func == 'mod_Rotated_Hyper_Ellipsoid':
            val = self.mod_Rotated_Hyper_Ellipsoid(x)
        return val
    
    def Ackley(self, x):
        a = 20
        b = 0.2
        c = 2*np.pi
        func = a + np.exp(1)
        sum1, sum2 = 0,0
        for i in range(self.dim):
            sum1 += x[i]**2
            sum2 += np.cos(c*x[i])
        func -= a * np.exp(-b * np.sqrt(sum1 / self.dim))
        func -= np.exp(sum2 / self.dim)
        
        func += 1
        
        func = np.log(func)
        return func, func
    
    def mod_Ackley(self, x):
        a = 20
        b = 0.2
        c = 2*np.pi
        func = a + np.exp(1)
        sum1, sum2 = 0,0
        for i in range(self.dim):
            sum1 += x[i]**2
            sum2 += np.cos(c*x[i])
        func -= a * np.exp(-b * np.sqrt(sum1 / self.dim))
        func -= np.exp(sum2 / self.dim)
        func += 1
        func = np.log(func)
        
        delta_log = self.delta_log
        true_func = func
        if true_func <= (self.best+delta_log):
            func = delta_log + (delta_log - true_func) 
        return func, true_func
        
    def Rastrigin(self, x):
        func = 10 * self.dim
        for i in range(self.dim):
            func += x[i]**2 - 10 * np.cos(2 * np.pi * x[i])
        return func, func

    def mod_Rastrigin(self, x):
        func = 10 * self.dim
        for i in range(self.dim):
            func += x[i]**2 - 10 * np.cos(2 * np.pi * x[i])
        delta_log = self.delta_log
        true_func = func
        if true_func <= (self.best+delta_log):
            func = delta_log + (delta_log - true_func) 
        return func, true_func
        
    def Levy(self, x):
        w = []
        for i in range(self.dim):
            w.append(1 + (x[i]-1)/4)
        term1 = (np.sin(np.pi * w[0]))**2
        term_sum = 0
        for i in range(self.dim-1):
            term_sum += ((w[i] - 1)**2) * (1 + 10 * (np.sin(np.pi * w[i] + 1))**2)
        
        term_end = ((w[self.dim-1] - 1)**2) * (1 + (np.sin(2 * np.pi * w[self.dim-1])**2))
        
        func = term1 + term_sum + term_end
        return func, func
        
    def mod_Levy(self, x):
        w = []
        for i in range(self.dim):
            w.append(1 + (x[i]-1)/4)
        term1 = (np.sin(np.pi * w[0]))**2
        term_sum = 0

        for i in range(self.dim-1):
            term_sum += ((w[i] - 1)**2) * (1 + 10 * (np.sin(np.pi * w[i] + 1))**2)
        
        term_end = ((w[self.dim-1] - 1)**2) * (1 + (np.sin(2 * np.pi * w[self.dim-1])**2))
        func = term1 + term_sum + term_end
        
        delta_log = self.delta_log
        true_func = func
        if true_func <= (self.best+delta_log):
            func = delta_log + (delta_log - true_func) 
        return func, true_func
        
    #x mellom 0, 1, 
    def Hartman_3D(self, x):
        a_matrix = [[3, 10, 30],
                    [0.1, 10, 35],
                    [3, 10, 30],
                    [0.1, 10, 35]]
        alfa = [1, 1.2, 3, 3.2]
        p_matrix = [[0.3689, 0.1170, 0.2673],
                    [0.4699, 0.4387, 0.7470],
                    [0.1091, 0.8732, 0.5547],
                    [0.03815, 0.5743, 0.8828]]

        func = 0
        for i in range(4):
            prod = 0
            for j in range(self.dim):
                prod += a_matrix[i][j] * (x[j] - p_matrix[i][j])**2
            func -= alfa[i] * np.exp(-prod)
        
        if len(x)==3:
            func += 3.86278
        
        return func, func
    
    
    def mod_Hartman_3D(self, x):
        a_matrix = [[3, 10, 30],
                    [0.1, 10, 35],
                    [3, 10, 30],
                    [0.1, 10, 35]]
        alfa = [1, 1.2, 3, 3.2]
        p_matrix = [[0.3689, 0.1170, 0.2673],
                    [0.4699, 0.4387, 0.7470],
                    [0.1091, 0.8732, 0.5547],
                    [0.03815, 0.5743, 0.8828]]

        func = 0
        for i in range(4):
            prod = 0
            for j in range(self.dim):
                prod += a_matrix[i][j] * (x[j] - p_matrix[i][j])**2
            func -= alfa[i] * np.exp(-prod)
        
        if len(x)==3:
            func += 3.86278
        
        delta_log = self.delta_log
        true_func = func
        if true_func <= (self.best+delta_log):
            func = delta_log + (delta_log - true_func) 
        return func, true_func
        
        
    #mellom 0 og pi, minima varierer, men 3D :  -1.8013
    def Michalewicz(self, x):
        m = 10
        func = 0
        for i in range(self.dim):
            func -= np.sin(x[i]) * (np.sin(((i+1) * x[i]**2) / np.pi))**(2*m)
        if len(x)==3:
            func += 1.8013
        return func, func
    
    def mod_Michalewicz(self, x):
        m = 10
        func = 0
        for i in range(self.dim):
            func -= np.sin(x[i]) * (np.sin(((i+1) * x[i]**2) / np.pi))**(2*m)
        if len(x)==3:
            func += 1.8013
            
        delta_log = self.delta_log
        true_func = func
        if true_func <= (self.best+delta_log):
            func = delta_log + (delta_log - true_func) 
        return func, true_func
        
    #mellom -50 og 50
    def Rotated_Hyper_Ellipsoid(self, x):
        func = 0
        for i in range(self.dim):
            for j in range(i+1):
                func += x[j]**2
        func += 1
        func = np.log(func)
        return func, func
        
    def mod_Rotated_Hyper_Ellipsoid(self, x):
        func = 0
        for i in range(self.dim):
            for j in range(i+1):
                func += x[j]**2

        delta_log = self.delta_log
        true_func = func
        if true_func <= (self.best+delta_log):
            func = delta_log + (delta_log - true_func)             
        func += 1
        func = np.log(func)
        return func, true_func
    
    def Rosenbrock(self, x):
        func = 0
        for i in range(self.dim-1):
            func += 100*(x[i+1]-x[i]**2)**2 + (1 - x[i])**2
        
        return func, func
    
    def mod_rosenbrock(self, x):
        func = 0
        delta_log = self.delta_log
        for i in range(self.dim-1):
            func += 100*(x[i+1]-x[i]**2)**2 + (1 - x[i])**2
        
        true_func = func
        if true_func <= (self.best+delta_log):
            func = delta_log + (delta_log - true_func)             
        return func, true_func
          
    def Eggholder(self, x):
        func = 0
        for i in range(self.dim-1):
            func -= (x[i+1]+47)*np.sin(np.sqrt(abs(x[i+1]+(x[i]/2)+47)))+ x[i]*np.sin(np.sqrt(abs(x[i]-(x[i+1]+47))))
        
        if len(x) == 2:
            func += 959.6407
        if len(x) == 3:
            func += 1597.278
        if len(x) == 4:
            func += 4043.574
        if len(x) == 5:
            func += 15292.97
        if len(x) == 6:
            func += 36524.74
        return func, func
    
    def mod_eggholder(self, x):
        func = 0
        delta_log = self.delta_log
        for i in range(self.dim-1):
            func -= (x[i+1]+47)*np.sin(np.sqrt(abs(x[i+1]+(x[i]/2)+47)))+ x[i]*np.sin(np.sqrt(abs(x[i]-(x[i+1]+47))))    

        if len(x) == 2:
            func += 959.6407
        if len(x) == 3:
            func += 1597.278
        if len(x) == 4:
            func += 4043.574
        if len(x) == 5:
            func += 15292.97
        if len(x) == 6:
            func += 36524.74
        true_func = func

        if true_func <= (self.best+delta_log):
            func = delta_log + (delta_log - true_func) 
        return func, true_func
    
    
    def Himmelblau(self, x):
        func = 0
        for i in range(self.dim-1):
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
        else:
            raise Exception("We don't know the minimum value for Himmelblau in this number of dimensions.")


        return func, func
    
    def mod_himmelblau(self, x):
        func = 0
        delta_log = self.delta_log
        for i in range(self.dim-1):
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

        true_func = func
        
        if true_func <= (self.best+delta_log):
            
            func = self.best + delta_log
            func = func  + delta_log * np.exp(- (self.best-(true_func))**2/(2*self.sigma**2)) 
        return func, true_func
    
    
    def mod_himmelblau2(self, x):
        func = 0
        delta_log = self.delta_log
        for i in range(self.dim-1):
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

        true_func = func
        
        if true_func <= (self.best+delta_log):
            func = delta_log + (delta_log - true_func) 
            
            # func = (true_func - delta_log)**2
            # print('fp',func, 'ft', true_func)    
    
        return func , true_func
        
        

