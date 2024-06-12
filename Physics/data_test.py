import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
path = r"C:/Users/Lenovo/Documents/GitHub/Master/Physics/eval_allggd_0305.csv"
# path = r"C:/Users/Lenovo/Documents/GitHub/Master/Physics/eval_data_1e5_shade_2904.csv"
# path = r"C:/Users/Lenovo/Documents/GitHub/Master/Physics/jde.csv"
# path = r"C:/Users/Lenovo/Documents/GitHub/Master/Physics/shade.csv"
# path = r"C:/Users/Lenovo/Documents/GitHub/Master/Physics/dshadepso.csv"

ar = pd.read_csv(path)
ind_g = []
ind_q = []
ind_n = []
lik = []
for index, row in ar.iterrows():
    if np.isnan(row[3]) ==False:
        ind_g.append(row[0])
        ind_q.append(row[2])
        ind_n.append(row[1])
        lik.append(row[3])
    
lik = np.array(lik)
print(np.min(lik))
a = np.argsort((-1) * lik)
lik = lik[a]
ind_g = np.array(ind_g)
ind_n = np.array(ind_n)
ind_q = np.array(ind_q)
ind_g = ind_g[a]
ind_n = ind_n[a]
ind_q = ind_q[a]
cmap = 'viridis_r'
vmin = np.min(lik)

fig, axs = plt.subplots(3,1, figsize = (15,25), tight_layout = 'True')
for ax in axs.flatten():
    ax.tick_params(axis='both', which='major', labelsize=15)
cmap = mpl.cm.viridis_r
# bounds = [-6, 2.7] 
norm = mpl.colors.Normalize(vmin = vmin,vmax =vmin+6) 

s1 = axs[0].scatter(ind_g, ind_q, c =lik, cmap = cmap, s = 1,norm = norm)
c1 = plt.colorbar(s1, ax=axs[0])        
c1.set_label('-$\Delta \ln \mathcal{L}$ ', fontsize=20)
c1.ax.tick_params(labelsize=15)

axs[0].set_xlabel('$m_g$ [GeV]', fontsize = '20')
axs[0].set_ylabel('$m_{\~q}$ [GeV]', fontsize = '20')
axs[0].set_facecolor('0')
axs[0].set_xlim(0,3000)
axs[0].set_ylim(0,3000)


s2 = axs[1].scatter(ind_g, ind_n, c =lik, cmap = cmap, s = 1, norm = norm)
c2 = plt.colorbar(s2, ax = axs[1])
c2.set_label('-$\Delta \ln \mathcal{L}$ ', fontsize=20)
c2.ax.tick_params(labelsize=15)

axs[1].set_facecolor('0')
axs[1].set_xlabel('$m_g$ [GeV]', fontsize = '20')
axs[1].set_ylabel('$m_\chi$ [GeV]' , fontsize = '20')
axs[1].set_xlim(0  ,3000)
axs[1].set_ylim(0  ,3000)

s3 = axs[2].scatter(ind_q, ind_n, c =lik, cmap = cmap, s = 1, norm=norm)
c3 = plt.colorbar(s3, ax=axs[2])
c3.set_label('-$\Delta \ln \mathcal{L}$ ', fontsize=20)
c3.ax.tick_params(labelsize=15)

axs[2].set_facecolor('0')
axs[2].set_xlabel('$m_{\~q}$ [GeV]', fontsize = '20')
axs[2].set_ylabel('$m_\chi$ [GeV]', fontsize = '20')
axs[2].set_xlim(0,3000)
axs[2].set_ylim(0,3000)
plt.savefig(r"C:/Users/Lenovo/Documents/GitHub/Master/Physics/allggd_phys.png")
# plt.show()