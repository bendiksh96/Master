from scipy.interpolate import LinearNDInterpolator
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import xsec

li =[1464.9863337865384,689.8915458869388,2233.4387540595844,-3.1356248208122253,11.922317356513918,8.617579239858578]

path_acc = r"C:\Users\Lenovo\Documents\GitHub\Master\Physics\HEPdata\Sig_acc_1.csv"
path_eff = r"C:\Users\Lenovo\Documents\GitHub\Master\Physics\HEPdata\Sig_eff_1.csv"
xa, ya, za = np.loadtxt(path_acc, comments='#', delimiter=',', unpack=True)
xe, ye, ze = np.loadtxt(path_eff, comments='#', delimiter=',', unpack=True)

interp_acc = LinearNDInterpolator(list(zip(xa, ya)), za, fill_value=np.nan)
interp_eff = LinearNDInterpolator(list(zip(xe, ye)), ze, fill_value=np.nan)

# x = np.linspace(500,3000,30)
# y = np.linspace(0,2000,20)
# X, Y = np.meshgrid(x,y)
# Z = interp_eff(X,Y)
# plt.pcolormesh(X,Y,Z)
# plt.colorbar()
# plt.show()


xsec.init(data_dir="gprocs")
xsec.set_energy(13000)
processes = [(1000021, 1000021)]
xsec.load_processes(processes)       
a = interp_acc.__call__(li[0:2])
b = interp_eff.__call__(li[0:2])
print(a)
print(b)
acc, eff = a,b
eff = eff/100
acc = acc/100
    
L = 139
squark_mass = li[2]
gluino_mass = li[0]
xsec.set_all_squark_masses(squark_mass)
xsec.set_gluino_mass(gluino_mass)
xsec.get_parameters()

res = xsec.eval_xsection(verbose = 0)
s = L * res[0] * acc[0] * eff[0]
s = s[0]
print(s)
xsec.clear_parameter(gluino_mass)
xsec.clear_parameter(squark_mass)
s = 15
n = 45 
b = 30

fac_n = float(math.factorial(n))
background  = n * np.log(b) - b - np.log(float(fac_n))
pred        = n*np.log(s+b) - (s + b) - np.log(float(fac_n))

target  = -( pred - background)
print('xsec:',res[0], 'signal', s,'tar.gz', target)
"""
nan:
509.6091930989112,456.46040055779827,858.7821261816625,nan,1
610.8675498670134,546.055050067709,2096.830781351713,nan,1
547.5904766083316,491.65036344443274,1484.2305112530125,nan,1
500.57370701464924,450.2267329352767,2464.2700644463935,nan,1
501.13111556215165,400.1638266147679,1236.7247769805085,nan,1
502.0862574210844,450.4244418418064,2482.1895055478235,nan,1
511.74497929312884,458.5487252770222,2558.712145110789,nan,1
501.4748757408764,408.6741554026756,2562.343931361085,nan,1
543.1613874448028,485.76119108465014,2242.540198210299,nan,1
502.7752486846304,428.4337078142725,1888.691265859556,nan,1
503.72287953679006,422.3172440255605,2838.0632921470533,nan,1


"""