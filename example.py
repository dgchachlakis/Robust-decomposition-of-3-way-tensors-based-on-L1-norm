import numpy as np
import matplotlib.pyplot as plt
import algorithm as l1tucker2
from scipy import linalg
from utils import *
D = 10
M = 10
N = 10
number_of_components = 4

data_tensor = np.random.randn(D, M, N)
U, V, metric_evolution = l1tucker2.am_l1_tucker2(data_tensor, number_of_components)

plt.figure()
plt.plot(metric_evolution)
plt.xlabel('Iteration index')
plt.ylabel('Metric')
plt.show()


#U = linalg.orth(np.random.randn(D, K))
#V = linalg.orth(np.random.randn(M, K))
