import numpy as np
def mysign(x):
    x = np.sign(x)
    x[np.where(x == 0)] = 1
    return x