import numpy as np
from itertools import accumulate

def cumulative(u, prepend_zero=False):
    v = list(accumulate(u))
    if( prepend_zero ): v.insert(0,0)
    return np.array(v) / v[-1]

def quantile(U, p, dt, ot):
    Q = np.zeros(len(p))
    q = 0
    t = 0
    for pp in p:
        if( t >= len(U) - 1 ): Q[q] = ot + (len(U) - 1) * dt
        #elif( U[t] == 0.0 ): Q[q] = ot
        else:
            while( U[t+1] < pp and t < len(U) ): t += 1
            if( t == len(U) - 1 ): Q[q] = (len(U) - 1)*dt
            else:
                Q[q] = ot + dt * (t + (pp-U[t]) / (U[t+1] - U[t]))
        q += 1
    return Q

def transport_distance(d,u,dt,ot):
    T = (len(u)-1)*dt
    t = np.linspace(0,T,len(u))
    D = cumulative(d)
    U = cumulative(u)
    Q = quantile(D,U,dt,ot)
    print('\n'.join([str(e) for e in D-U]))
    return Q - t 

def wass_adjoint(**kw):
    Q = kw.get('Q', None)
    if( Q == None ):
        Q = transport_distance(kw['d'], kw['u'], kw['dt'], kw['ot'])
    integral = np.flip(list(accumulate(np.flip(Q))))
    return kw['dt'] * integral, Q

def wass_adjoint_and_eval(**kw):
    if( 'multi' in kw.keys() ):
        evaluations = []
        adjoints = []
        transports = []
        Q = kw.get('Q', None)
        assert( Q == None )
        assert( len(kw['u']) == len(kw['d']) )
        assert( type(kw['dt']) == float )
        for (dd,uu) in zip(kw['u'], kw['d']):
            curr_eval, curr_adj, curr_Q = wass_adjoint_and_eval(d=dd, u=uu, 
                dt=kw['dt'])
            evaluations.append(curr_eval)
            adjoints.append(curr_adj)
            transports.append(curr_Q)
        return np.array(evaluations), np.array(adjoints), np.array(transports)
    else:
        adj, Q = wass_adjoint(**kw)
        return adj[-1]**2 / kw['dt'], adj, Q
    