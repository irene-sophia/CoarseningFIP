import numpy as np

def ismember1D(A, Bs, sortInds):
    # B is a 1D vector which does not need to be unique and A is a single value
    _, firstInds = np.isin(A, Bs, assume_unique=False, return_indices=True)
    _, lastInds = np.isin(A, Bs, assume_unique=False, return_indices=True)
    
    if (firstInds == 0) or (lastInds == 0):
        index = []
    else:
        index = sortInds[firstInds:lastInds]
    
    return index