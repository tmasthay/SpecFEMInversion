import numpy as np
from itertools import accumulate

def cumulative(u):
    v = np.array(list(accumulate(u)))
    return v / v[-1]

def quantile(U, dp, dt):
    curr = 0
    Q = np.zeros(int(np.ceil(1/dp + 1)))
    p = 0.0
    inv_dp = 1.0 / dp
    q = 1
    t = 0
    while( p <= 1.0 ):
        if( t >= len(U) - 1 ): Q[q] = (len(U) - 1) * dt
        else:
            while( U[t+1] < p and t < len(U) ): t += 1
            if( t == len(U) - 1 ): Q[q] = (len(U) - 1)*dt
            else:
                Q[q] = dt * (t + (p-U[t]) / (U[t+1] - U[t]))
        q += 1
        p += dp
    return Q