import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = r"C:/Users/Lenovo/Documents/GitHub/Master/Physics/eval_data_init.csv"

ar = pd.read_csv(path)
ind_g = []
ind_q = []
ind_n = []
lik = []
for index, row in ar.iterrows():
    ind_g.append(row[0])
    ind_q.append(row[1])
    ind_n.append(row[2])
    lik.append(row[3])
    
lik = np.array(lik)
arg = np.where(lik<10)
lik = lik[arg]
a = np.argsort((-1)*lik)
lik = lik[a]
ind_g = np.array(ind_g)
ind_n = np.array(ind_n)
ind_q = np.array(ind_q)
ind_g = ind_g[arg]
ind_q = ind_q[arg]
ind_n = ind_n[arg]
cmap = 'viridis_r'
norm = plt.Normalize(lik.min(), lik.max())


fig, axs = plt.subplots(2,1, figsize = (15,10), tight_layout = 'True')
fig.set_facecolor('ivory')
s1 = axs[0].scatter(ind_g, ind_q, c =lik, cmap = cmap, norm = norm)
c1 = plt.colorbar(s1, ax=axs[0])    
axs[0].set_xlabel('gluino_mass')
axs[0].set_ylabel('quark_mass')
axs[0].set_facecolor('#36013f')
axs[0].set_xlim(500,3000)
axs[0].set_ylim(0,2000)


s2 = axs[1].scatter(ind_g, ind_n, c =lik, cmap = cmap, norm = norm)
c2 = plt.colorbar(s2, ax=axs[1])
axs[1].set_xlabel('gluino_mass')
axs[1].set_ylabel('neutralino_mass')
axs[1].set_facecolor('#36013f')
axs[1].set_xlim(500,3000)
axs[1].set_ylim(500,3000)

# s3 = axs[2].scatter(ind_n, ind_q, c =lik, cmap = cmap, norm = norm)
# c3 = plt.colorbar(s3, ax=axs[2])
# axs[2].set_xlabel('quark_mass')
# axs[2].set_ylabel('neutralino_mass')
# axs[2].set_facecolor('#36013f')
# axs[2].set_xlim(500,3000)
# axs[2].set_ylim(0,2000)

plt.show()