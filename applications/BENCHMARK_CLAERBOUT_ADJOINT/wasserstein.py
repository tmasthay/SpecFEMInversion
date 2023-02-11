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

def transport_distance(d,u,dt):
    T = (len(u)-1)*dt
    t = np.linspace(0,T,len(u))
    return quantile(cumulative(d), cumulative(u), dt) - t 

def wass_adjoint(**kw):
    Q = kw.get('Q', None)
    if( Q == None ):
        Q = transport_distance(kw['d'], kw['u'], kw['dt'])
    return kw['dt'] * np.array(list(accumulate(Q))), Q

def wass_adjoint_and_eval(**kw):
    adj, Q = wass_adjoint(**kw)
    return adj[-1]**2 / kw['dt'], adj, Q
    