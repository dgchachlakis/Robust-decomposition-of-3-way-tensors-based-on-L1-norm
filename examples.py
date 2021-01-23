import numpy as np
import matplotlib.pyplot as plt
import algorithm as l1tucker2
from scipy import linalg
from utils import *
D = 20
M = 25
N = 20
number_of_components = 8
data_tensor = np.random.randn(D, M, N)
left = linalg.orth(np.random.randn(D, number_of_components))
right = linalg.orth(np.random.randn(M, number_of_components))
# ===========================
# Example 1 -- AM-L1-TUCKER2 with no user-defined initialization
U, V, metric_evolution1 = l1tucker2.am_l1_tucker2(data_tensor, number_of_components)
plt.figure()
plt.plot(metric_evolution1)
plt.xlabel('Iteration index')
plt.ylabel('Metric')
plt.show()
# ===========================
# Example 2 -- AM-L1-TUCKER2 with user-defined initialization of the left-side factor
U, V, metric_evolution2 = l1tucker2.am_l1_tucker2(data_tensor, number_of_components, left)
plt.figure()
plt.plot(metric_evolution2)
plt.xlabel('Iteration index')
plt.ylabel('Metric')
plt.show()
# ===========================
# Example 3 -- AM-L1-TUCKER2 with user-defined initialization of both factors
U, V, metric_evolution3 = l1tucker2.am_l1_tucker2(data_tensor, number_of_components, left, right)
plt.figure()
plt.plot(metric_evolution3)
plt.xlabel('Iteration index')
plt.ylabel('Metric')
plt.show()
# ===========================