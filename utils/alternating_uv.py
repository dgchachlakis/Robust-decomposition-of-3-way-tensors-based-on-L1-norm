import numpy as np
from .procrustes import procrustes
def alternating_uv(Z, right_factor, tolerance = 1e-6):
    (M, K) = right_factor.shape
    D = int(Z.shape[0] / K)
    metric_evolution = list()
    while True:
        Rv = np.reshape(Z @ right_factor.flatten('F'), (D, K), order = 'F')
        left_factor = procrustes(Rv)
        metric_evolution.append(metric_eval(left_factor, right_factor, Z))
        Ru = np.reshape(Z.T @ left_factor.flatten('F'), (M, K), order = 'F')
        right_factor = procrustes(Ru)
        metric_evolution.append(metric_eval(left_factor, right_factor, Z))
        if metric_evolution[-1] - metric_evolution[-2] <= tolerance: 
            break
    return left_factor, right_factor, metric_evolution
def metric_eval(left_f, right_f, Z):
    return left_f.flatten('F').T @ Z @ right_f.flatten('F')