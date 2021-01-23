import numpy as np
def zofb(data_tensor, aux_bin_tensor):
    (D, M, N) = data_tensor.shape
    K = aux_bin_tensor.shape[0]
    Z = np.zeros((D * K, M * K))
    for i in range(K):
        idr = np.array(range(i * D, i * D + D))
        for k in range(K):
            idc = np.array(range(k * M, k * M + M))
            b = aux_bin_tensor[i, k, :].flatten()
            b2 = np.kron(b, np.ones(D * M, )).copy()
            B = b2.reshape((D, M, N), order = 'F')
            Z[np.ix_(idr, idc)] = np.sum(B * data_tensor, axis=2)
    return Z
    