import numpy as np
def procrustes(matrix):
    O, S, Vt = np.linalg.svd(matrix, full_matrices = False)
    return O @ Vt
