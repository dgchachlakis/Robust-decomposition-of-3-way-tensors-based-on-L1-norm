import numpy as np
def update_z(data_tensor, Z, Bnew, Bold):
    (D, M, N) = data_tensor.shape
    K = Bold.shape[0]
    dif = Bnew != Bold
    kax_out = np.where(np.sum(np.sum(dif, axis = 2), axis = 1) >= 0)[0]
    for ko in kax_out:
        difin = np.reshape(dif[ko, :, :], (K, N))
        kax_in = np.where(np.sum(difin, axis = 1) > 0)[0]
        print(kax_in)
        for ki in kax_in:
            difinin = np.reshape(difin[ki, :], (1, N))
            idx = np.where(difinin > 0)[0]
            if idx.shape[0] > 0:
                idr = range(ki * D, ki * D + D)
                idc = range(ko * M, ko * M + M)
                Y = Z[np.ix_(idr, idc)]
                for i in range(idx.shape[0]):
                    newbit = Bnew[ki, ko, idx[i]]
                    Y = Y + 2 * newbit * data_tensor[:, :, idx[i]]
                Z[np.ix_(idr, idc)] = Y
    return Z