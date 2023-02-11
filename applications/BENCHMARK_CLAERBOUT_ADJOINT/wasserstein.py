import numpy as np
from itertools import accumulate

def cumulative(u):
    v = np.array(list(accumulate(u)))
    return v / v[-1]