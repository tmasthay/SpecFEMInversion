import numpy as np
from itertools import accumulate
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.stats.sampling import NumericalInverseHermite

def cts_quantile(x,y, kind='cubic', resolution=1e-5):
    class Dummy:
        def __init__(self, pdf_arg, cdf_arg):
            self.pdf = pdf_arg
            self.cdf = cdf_arg
    dx = x[1] - x[0]
    p = interp1d(x,y,kind=kind)
    c = interp1d(x,cumulative_trapezoid(y,dx=dx,initial=0.0))
    f = Dummy(p,c)
    order = 3 if kind == 'cubic' else 5 if kind == 'quintic' else 1
    f.ppf = NumericalInverseHermite(
        f, 
        domain=(x[0],x[-1]),
        order=order,
        u_resolution=resolution
    ).ppf
    return f

def split_normalize(f):
    pos = np.array([e if e > 0 else 0.0 for e in f])
    neg = np.array([-e if e < 0 else 0.0 for e in f])
    C1 = sum(pos)
    C2 = sum(neg)
    pos = pos if C1 == 0.0 else pos / C1
    neg = neg if C2 == 0.0 else neg / C2
    return pos, neg

def cut(n,restrict):
    if( type(restrict) == type(None) ):
        return 0,n
    if( type(restrict) == float or len(restrict) == 1 ):
        if( restrict > 0.5 ): restrict = 1.0 - restrict
        restrict = [restrict, 1.0 - restrict]
    p,q = restrict
    assert( p <= q and p <= 1.0 and q <= 1.0 )
    i1 = int(p * n)
    i2 = int(q * n) + 1
    if( i2 > n ): i2 = n
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

def wass_distance(d,u,dt,ot,**kw):
    nt = len(d)
    restrict = kw.get('restrict', None)
    i1, i2 = cut(nt, restrict)
    Q,D,U = transport_distance(d,u,dt,ot)
    return dt * np.trapz(Q[i1:i2]**2*u[i1:i2])

def wass_poly(d,u,dt,ot, **kw):
    norm = kw.get('norm', split_normalize)
    d_norm = norm(d)
    u_norm = norm(u)
    total = 0.0
    for (dd,uu) in zip(d_norm, u_norm):
        total += wass_distance(dd,uu,dt,ot,**kw)
    return total

def accumulate_adjoint(g):
    u = np.flip(np.cumsum(np.flip(g)))
    return np.append(0.5 * (u[:-1] + u[1:] - g[-1]), 0.0)

def wass_adjoint(**kw):
    Q = kw.get('Q', None)
    if( Q == None ):
        Q, D, U = transport_distance(kw['d'], kw['u'], kw['dt'], kw['ot'])
    #integral = np.flip(list(accumulate(np.flip(Q))))
    i1, i2 = cut(len(Q), kw.get('restrict', None))
    Q,D,U = Q[i1:i2],D[i1:i2],U[i1:i2]
    adjoint = Q**2 + 2 * kw['dt'] * accumulate_adjoint(Q*kw['u'][i1:i2])
    return adjoint,Q,D,U

def wass_adjoint_and_eval(**kw):
    if( kw.get('multi', False) ):
        data = [ [], [], [], [], [] ]
        adjoints = []
        transports = []
        synthetic = []
        Q = kw.get('Q', None)
        assert( Q == None )
        assert( len(kw['u']) == len(kw['d']) )
        assert( type(kw['dt']) == float )
        for (dd,uu) in zip(kw['u'], kw['d']):
            curr = wass_adjoint_and_eval(d=dd, u=uu, 
                dt=kw['dt'], ot=kw['ot'], restrict=kw['restrict'])
            for i in range(len(data)):
                data[i].append(curr[i])
        data = [np.array(sum(e)) for e in data]
        return data[0], data[1], data[2], data[3], data[4]
    else:
        adj, Q, D, U = wass_adjoint(**kw)
        i1,i2 = cut(len(kw['u']), kw.get('restrict', None))
        return np.trapz(Q**2*kw['u'][i1:i2]) * kw['dt'], \
            adj, \
            Q, \
            D, \
            U
    
def wass_v2(g,x,kind='cubic',resolution=1e-5,store_q=True, restrict=None):
    dx = x[1] - x[0]
    dist_ref = cts_quantile(x,g,kind=kind,resolution=resolution)
    i1,i2 = cut(len(x), restrict)
    def helper(f, F):
        return np.trapz((dist_ref.ppf(F)[i1:i2] - x[i1:i2])**2*f[i1:i2], dx=dx)
    if( store_q ):
        return helper, dist_ref
    else:
        return helper
