import numpy as np

path    = (r"C:\Users\Lenovo\Documents\Master\Tests\3d_validation_rosenbrock.npy")
arr = np.load(path)

print(arr[60,60,:])
exit()


# path    = (r'C:\Users\Bendik Selvaag-Hagen\OneDrive - Universitetet i Oslo\Documents\GitHub\Master\datafile.csv')
ds      = pd.read_csv(path, delimiter=',',header = None)#, on_bad_lines='skip')

likelihood_list = []
individual_list = []
index_list      = []
for index, row in ds.iterrows(): 
    var = []
    for j in range(self.dim):     
        var.append(row[j])
    likelihood_list.append(row[j+1])
    individual_list.append(var)
    index_list.append(row[j+2])
    
self.likelihood     = np.array(likelihood_list)
# for i in range(len(self.likelihood)):
#     print(self.likelihood[i])
self.individuals    = np.array(individual_list)
self.index          = np.array(index_list)
print('Number of points:', len(self.likelihood))
