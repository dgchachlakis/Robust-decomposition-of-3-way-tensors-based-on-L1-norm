import numpy as np
def l1tucker2_metric(tensor, U, V):
    (D, M, N) = tensor.shape
    met = 0
    for n in range(N):
        met += np.sum(np.abs(U.T @ tensor[:, :, n] @ V))
    return met