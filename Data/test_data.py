import sys
import pandas as pd

path    = (r"C:\Users\Lenovo\Documents\Master\Data\super_data.txt")
# path    = (r'C:\Users\Bendik Selvaag-Hagen\OneDrive - Universitetet i Oslo\Documents\GitHub\Master\datafile.csv')
ds      = pd.read_csv(path, delimiter=',',header = None)#, on_bad_lines='skip')
c = 0
a = 100
for index, row in ds.iterrows():
    #Row[0] ~ Occupancy
    #Row[1] ~ Likelihood
    c += row[0]
    if row[1] < a:
        a = row[1]

print(c)
print(a)