import numpy as np
def mysign(x):
    x = np.sign(x)
    x[x == 0] = 1
    return x