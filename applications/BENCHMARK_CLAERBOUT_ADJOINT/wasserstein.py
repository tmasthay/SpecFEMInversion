import numpy as np
from itertools import accumulate

def cut(n,restrict):
    if( type(restrict) == None ):
        return 0,n
    if( type(restrict) == float ):
        restrict = [restrict, 1.0 - restrict]
    assert( p <= q and p <= 1.0 and q <= 1.0 )
    i1 = int(p * n)
    i2 = int(q * n) + 1
    return i1,i2

def cumulative(u, prepend_zero=False):
    v = list(accumulate(u))
    if( prepend_zero ): v.insert(0,0)
    return np.array(v) / v[-1]

def quantile(U, p, dt, ot, div_tol=1e-16, tail_tol=1e-30):
    Q = np.zeros(len(p))
    q = 0
    t = np.where(U > 0)[0][0]
    for pp in p:
        if( t >= len(U) - 1 or pp > 1-tail_tol ): 
            Q[q] = ot + (len(U) - 1) * dt
        elif( U[t] == 0.0 ): Q[q] = ot
        else:
            while( U[t+1] < pp and t < len(U) ): t += 1
            if( t == len(U) - 1 ): Q[q] = (len(U) - 1)*dt
            else:
                if( abs(U[t+1] - U[t]) >= div_tol ):
                    Q[q] = ot + dt * (t + (pp-U[t]) / (U[t+1] - U[t]))
                else:
                    Q[q] = ot + dt * t
        q += 1
    return Q

def transport_distance(d,u,dt,ot):
    T = ot + (len(u)-1)*dt
    t = np.linspace(ot,T,len(u))
    D = cumulative(d)
    U = cumulative(u)
    Q = quantile(D,U,dt,ot)
    return Q-t, D, U

def accumulate_adjoint(g):
    u = np.flip(np.cumsum(np.flip(g)))
    return np.append(0.5 * (u[:-1] + u[1:] - g[-1]), 0.0)

def wass_adjoint(**kw):
    Q = kw.get('Q', None)
    if( Q == None ):
        Q, D, U = transport_distance(kw['d'], kw['u'], kw['dt'], kw['ot'])
    #integral = np.flip(list(accumulate(np.flip(Q))))
    integral = kw['dt'] * accumulate_adjoint(Q)
    i1, i2 = cut(len(Q), kw.get('restrict', None))
    return integral[i1:i2],Q[i1:i2],D[i1:i2],U[i1:i2]

def wass_adjoint_and_eval(**kw):
    restrict = kw.get('restrict', None)
    if( 'multi' in kw.keys() ):
        data = [ [], [], [], [], [] ]
        adjoints = []
        transports = []
        data = []
        synthetic = []
        Q = kw.get('Q', None)
        assert( Q == None )
        assert( len(kw['u']) == len(kw['d']) )
        assert( type(kw['dt']) == float )
        for (dd,uu) in zip(kw['u'], kw['d']):
            curr = wass_adjoint_and_eval(d=dd, u=uu, 
                dt=kw['dt'])
            for i in range(len(data)):
                data[i].append(curr[i])
        return [np.array(e) for e in data]
    else:
        adj, Q, D, U = wass_adjoint(**kw)
        i1,i2 = cut(len(Q), restrict)
        return np.trapz(Q[i1:i2]**2) * kw['dt'], \
            adj[i1:i2], \
            Q[i1:i2], \
            D[i1:i2], \
            U