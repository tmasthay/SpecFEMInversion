import numpy as np
from itertools import accumulate

def cumulative(u):
    v = np.array(list(accumulate(u)))
    return v / v[-1]

def quantile(U, p, dt):
    Q = np.zeros(int(np.ceil(1.0/dp + 1)))
    q = 1
    t = 0
    for pp in p:
        if( t >= len(U) - 1 ): Q[q] = (len(U) - 1) * dt
        else:
            while( U[t+1] < pp and t < len(U) ): t += 1
            if( t == len(U) - 1 ): Q[q] = (len(U) - 1)*dt
            else:
                Q[q] = dt * (t + (p-U[t]) / (U[t+1] - U[t]))
        q += 1
    return Q

def wass_adjoint(u,d,dt):
    alpha = 0.9
    du = []
    D = quantile