import numpy as np
from itertools import accumulate

def cumulative(u):
    v = list(accumulate(u))
    v.insert(0,0)
    return np.array(v) / v[-1]

def quantile(U, p, dt):
    Q = np.zeros(len(p))
    q = 0
    t = 0
    for pp in p:
        if( t >= len(U) - 1 ): Q[q] = (len(U) - 1) * dt
        elif( U[t] == 0.0 ): Q[q] = 0.0
        else:
            while( U[t+1] < pp and t < len(U) ): t += 1
            if( t == len(U) - 1 ): Q[q] = (len(U) - 1)*dt
            else:
                Q[q] = dt * (t + (pp-U[t]) / (U[t+1] - U[t]))
        q += 1
    return Q

def wass_adjoint(u,d,dt):
    Q = quantile(cumulative(d), cumulative(u), dt)
    T = (len(u)-1)*dt
    t = np.linspace(0,T,len(u))
    return T**2 - t**2 + np.array(list(accumulate(Q)))