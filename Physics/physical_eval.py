from scipy.interpolate import LinearNDInterpolator
import numpy as np
import matplotlib.pyplot as plt
import sys
import xsec

class XSection:
    def __init__(self):    
        xsec.init(data_dir="gprocs")
        xsec.set_energy(13000)
        # Load GP models for the specified process gg-> n
        processes = [(1000021, 1000021)]
        xsec.load_processes(processes)

    def eval_cross_section(self, ind):
        gluino_mass = ind[0]
        neutralino_mass = ind[1]
        squark_mass = ind[2]
        
        xsec.set_all_squark_masses(squark_mass)
        xsec.set_gluino_mass(gluino_mass)
        xsec.get_parameters()
        res = xsec.eval_xsection(verbose = 0)
        xsec.clear_parameter(gluino_mass)
        xsec.clear_parameter(squark_mass)
        print(res[0])
xs = XSection()
mg = np.linspace(500,2000, 10)
mq = np.linspace(500,2000, 10)
for i in range(9):
    nu = [mg[i], 100, mq[i]]
    xs.eval_cross_section(nu)
    



path_acc = r"C:\Users\Lenovo\Documents\GitHub\Master\Physics\HEPdata\Sig_acc_1.csv"
path_eff = r"C:\Users\Lenovo\Documents\GitHub\Master\Physics\HEPdata\Sig_eff_1.csv"
xa, ya, za = np.loadtxt(path_acc, comments='#', delimiter=',', unpack=True)
xe, ye, ze = np.loadtxt(path_eff, comments='#', delimiter=',', unpack=True)
interp_acc = LinearNDInterpolator(list(zip(xa, ya)), za, fill_value=np.nan)
interp_eff = LinearNDInterpolator(list(zip(xe, ye)), ze, fill_value=np.nan)

print(interp_eff.__call__(1200,800))
