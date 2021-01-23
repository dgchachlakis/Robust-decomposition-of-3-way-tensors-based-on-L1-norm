import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from utils import *

D = 5
M = 5
N = 10
K = 3
print('\n'*5)
U=linalg.orth(np.random.randn(D, K))
V=linalg.orth(np.random.randn(M, K))
X=np.random.randn(D, M, N)
B=update_b(X, U, V)
Z=zofb(X, B)

un, v, met = alternating_uv(Z, V)

B2 = np.sign(np.random.randn(K, K, N))

ZZ = update_z(X, Z, B2, B)

plt.figure()
plt.plot(met)
plt.show()