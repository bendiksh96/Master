import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = r'C:\Users\Lenovo\Documents\GitHub\Master\Physics\eval_allggd_0305.csv'
data = pd.read_csv(path)

data_arr = np.zeros((int(1e5),4))

for index, row in data.iterrows():
    data_arr[index, 0 ] = row[0] 
    data_arr[index, 1 ] = row[1] 
    data_arr[index, 2 ] = row[2] 
    data_arr[index, 3 ] = row[3] 

res =  40
bin_g = np.linspace(0,3000, res)
bin_n = np.linspace(0,3000, res)
bin_q = np.linspace(0,3000, res)

blompi =  np.ones((res,res))*1000

for j in range(index):
    for g in range(len(bin_g)-1):
        for n in range(len(bin_n)-1):
            if data_arr[j, 0] > bin_g[g] and  data_arr[j, 0] < bin_g[g+1]:
                if data_arr[j, 1] > bin_n[n] and  data_arr[j, 1] < bin_n[n+1]:
                    if data_arr[j, -1] < blompi[g,n]:
                        blompi[g,n] = data_arr[j, -1]


# plt.imshow(blompi)
plt.pcolormesh(bin_g, bin_n, blompi)
plt.colorbar()
plt.show()