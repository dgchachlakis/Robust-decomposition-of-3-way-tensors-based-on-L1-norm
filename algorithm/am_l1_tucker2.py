import numpy as np
from scipy import linalg
from utils import *
def am_l1_tucker2(data_tensor, number_of_components, tolerance = 1e-6, init_factors = None):
    (D, M, N) = data_tensor.shape
    if init_factors != None:
        (left_factor, right_factor) = init_factors
    else:
        left_factor = linalg.orth(np.random.randn(D, number_of_components))
        right_factor = linalg.orth(np.random.randn(M, number_of_components))
    aux_bin_tensor_old = update_b(data_tensor, left_factor, right_factor)
    Z_old = zofb(data_tensor, aux_bin_tensor_old)
    metric_evolution = [metric_eval(left_factor, right_factor, Z_old)]
    while True:
        if number_of_components == 1:
            u, s, vt = np.linalg.svd(Z_old, full_matrices = False)
            left_factor, right_factor = u[:, 0], vt.T[:,0]
        else:
            left_factor, right_factor, metev = alternating_uv(Z_old, right_factor)
        aux_bin_tensor_new = update_b(data_tensor, left_factor, right_factor)
        Z_new = update_z(data_tensor, Z_old, aux_bin_tensor_new, aux_bin_tensor_old)
        # Z_new = zofb(data_tensor, aux_bin_tensor_new)
        metric_evolution.append(metric_eval(left_factor, right_factor, Z_new))
        if metric_evolution[-1] - metric_evolution[-2] <= tolerance:
            break
        aux_bin_tensor_old = aux_bin_tensor_new
        Z_old = Z_new
    return left_factor, right_factor, metric_evolution[1:]    
def metric_eval(left_f, right_f, Z):
    return left_f.flatten('F').T @ Z @ right_f.flatten('F')   