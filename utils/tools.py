import torch
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def norm_data(loader,location=None):
    if location is not None:
        for feature, status, time in loader:
            x = scaler.fit_transform(np.log(feature[:, location] + 1))
            time = time
            status = status
    else:
        for feature, status, time in loader:
            x = scaler.fit_transform(np.log(feature + 1))
            time = time
            status = status
    return x, time, status
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def CIndex(pred, ytime_test, ystatus_test):
    concord = 0.
    total = 0.
    N_test = ystatus_test.shape[0]
    ystatus_test = np.asarray(ystatus_test, dtype=bool)
    theta = pred
    for i in range(N_test):
        if ystatus_test[i] == 1:
            for j in range(N_test):
                if ytime_test[j] > ytime_test[i]:
                    total = total + 1
                    if theta[j] < theta[i]: concord = concord + 1
                    elif theta[j] == theta[i]: concord = concord + 0.5
    return(concord/(total+1e-8))

