from scipy.interpolate import LinearNDInterpolator
import numpy as np
import matplotlib.pyplot as plt
import sys

x, y, z = np.loadtxt("HEPData-ins1827025-v2-Signal_acceptance_1.csv", comments='#', delimiter=',', unpack=True)

X = np.linspace(min(x), max(x), 41)
Y = np.linspace(min(y), max(y), 26)
X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
interp = LinearNDInterpolator(list(zip(x, y)), z, fill_value=np.nan)

# a = interp(2200, 1375)
# print(a)

Z = interp(X, Y)
print(Z.shape)
print(Z)


plt.pcolormesh(X, Y, Z, shading='auto')
plt.plot(x, y, "ok", label="input point")
plt.legend()
plt.colorbar()
plt.axis("equal")
plt.show()