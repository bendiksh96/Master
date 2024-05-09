import numpy as np

path = r"C:/Users/Lenovo/Documents/Master/Data/rastrigin_5D_40_result.npy"

arg = np.load(path)

print(np.min(arg))