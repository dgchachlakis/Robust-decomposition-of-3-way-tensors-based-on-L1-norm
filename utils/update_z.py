import numpy as np
def update_z(data_tensor, Zold, Bnew, Bold):
    Z = Zold
    (D, M, N) = data_tensor.shape
    K = Bold.shape[0]
    dif = (Bnew != Bold) * 1
    kax_out = np.where(np.sum(np.sum(dif, axis = 2), axis = 1) > 0)[0]
    if kax_out.size > 0 :
        for ko in np.nditer(kax_out):
            difin = np.squeeze(dif[ko, :, :])
            kax_in = np.where(np.sum(difin, axis = 1) > 0)[0]
            for ki in np.nditer(kax_in):
                difinin = difin[ki, :]
                idx = np.where(difinin > 0)[0]
                if idx.shape[0] > 0:
                    idr = np.array(range(ki * D, ki * D + D))
                    idc = np.array(range(ko * M, ko * M + M))
                    Y = Z[np.ix_(idr, idc)]
                    for i in np.nditer(idx):
                        newbit = Bnew[ki, ko, i]
                        Y = np.add(Y, 2 * newbit * data_tensor[:, :, i])
                    Z[np.ix_(idr, idc)] = Y
    return Z