import numpy as np
from .mysign import mysign
def update_b(tensor, left_factor, right_factor):
    (D, M, N) = tensor.shape
    K = left_factor.shape[1]
    B = np.zeros((K, K, N))
    for n in range(N):
        B[:, :, n] = mysign(left_factor.T @ tensor[:, :, n] @ right_factor)
    return B