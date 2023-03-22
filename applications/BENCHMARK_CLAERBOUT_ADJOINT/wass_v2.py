import numpy as np
from itertools import accumulate
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.stats.sampling import NumericalInverseHermite
from helper_functions import helper

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
    f_abs = np.abs(f)
    pos = 0.5 * (f_abs + f)
    neg = f - pos
    c_pos = np.sum(pos)
    c_neg = np.sum(neg)
    if( c_pos > 0 ):
        pos /= c_pos
    if( c_neg > 0 ):
        neg /= c_neg
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
    
def create_evaluators(t, path):
    hf = helper()
    data_x = hf.read_SU_file('%s/Ux_single_d.su'%path)
    data_z = hf.read_SU_file('%s/Uz_single_d.su'%path)
