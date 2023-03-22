import numpy as np
from itertools import accumulate
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.stats.sampling import NumericalInverseHermite
from helper_functions import helper
import time
import pickle
import matplotlib.pyplot as plt

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

def eff_support(x, tol):
    i1 = np.argmax(x >= tol)
    i2 = len(x) - 1 - np.argmax(x[::-1] >= tol)
    return i1,max(i2, i1+1)

def split_normalize(f, dx, clip_val=None):
    clip_val = 0.0 if type(clip_val) == None else clip_val
    f_abs = np.abs(f)
    pos = 0.5 * (f_abs + f)
    neg = pos - f
    pos = pos + clip_val
    neg = neg + clip_val

    c_pos = np.trapz(pos, dx=dx)
    c_neg = np.trapz(neg, dx=dx)
    if( c_pos > 0 ):
        pos /= c_pos
    if( c_neg > 0 ):
        neg /= c_neg
    # if( type(clip_val) != None ):
    #     i1,i2 = eff_support(pos, clip_val * np.max(pos))
    #     i3,i4 = eff_support(neg, clip_val * np.max(neg))
    #     return pos, neg, range(i1,i2), range(i3,i4)
    # else:
    #     return pos, neg
    return pos,neg
    
def square_normalize(f, dx, clip_val=None):
    u = f**2
    c = np.trapz(u,dx=dx)
    u = u if c == 0 else u / c
    if( type(clip_val) != None ):
        i1,i2 = eff_support(u, clip_val * np.max(u))
        return u, range(i1,i2)
    else:
        return u

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
    
def create_evaluators(
        t,
        input_path='convex_reference',
        output_path='evaluators',
        **kw
):
    hf = helper()
    kind = kw.get('kind', 'cubic')
    resolution = kw.get('resolution', 1e-5)
    data_x = hf.read_SU_file('%s/Ux_file_single_d.su'%input_path)
    data_z = hf.read_SU_file('%s/Uz_file_single_d.su'%input_path)
    evaluators = []
    dt = t[1] - t[0]
    start_time = time.time()
    num_recs = data_x.shape[0]
    tau = 0.01
    for i in range(num_recs):
        avg_time = (time.time() - start_time) / max(i,1)
        print('%d/%d ||| ELAPSED: %.2e ||| ETA: %.2e'%(
            i,
            data_x.shape[0],
            avg_time * max(i,1),
            avg_time * (num_recs - i)
            ),
            flush=True
        )
        ux_norm, ix = square_normalize(data_x[i], dt, clip_val=tau)
        uz_norm, iz = square_normalize(data_z[i], dt, clip_val=tau)
        if( i < 5 ):
            plt.subplot(1,3,1)
            plt.plot(t, data_x[i], label='rawx', color='blue')
            plt.plot(t, data_z[i], label='rawz', color='red', linestyle='-.')
            plt.legend()

            plt.subplot(1,3,2)
            plt.plot(t[ix], ux_norm[ix], label='x', linestyle='-', color='green')
            plt.plot(t[iz], uz_norm[iz], label='z', linestyle='-.', color='red')
            plt.legend()

            plt.subplot(1,3,3)
            plt.plot(
                t[ix],
                cumulative_trapezoid(ux_norm[ix], dx=dt, initial=0.0),
                label='Windowed CDF x', 
                linestyle='-', 
                color='green'
            )
            plt.plot(
                t[iz],
                cumulative_trapezoid(uz_norm[iz], dx=dt, initial=0.0),
                label='Windowed CDF z', 
                linestyle='-.', 
                color='red'
            )
            plt.savefig('%d.pdf'%i)
        wx = wass_v2(
            ux_norm[ix],
            t[ix],
            kind=kind,
            resolution=resolution,
            store_q=False,
            restrict=None
        )
        wz = wass_v2(
            uz_norm[iz],
            t[iz],
            kind=kind,
            resolution=resolution,
            store_q=False,
            restrict=None
        )
        # ux_pos, ux_neg, ixp, ixn = split_normalize(data_x[i], dt, clip_val=tau)
        # uz_pos, uz_neg, izp, izn = split_normalize(data_z[i], dt, clip_val=tau)
        # ux_pos, ux_neg = split_normalize(data_x[i], dt, clip_val=tau)
        # uz_pos, uz_neg = split_normalize(data_z[i], dt, clip_val=tau)


        # wx_pos = wass_v2(
        #     ux_pos,
        #     t,
        #     kind=kind,
        #     resolution=resolution,
        #     store_q=False,
        #     restrict=None
        # )
        # wx_neg = wass_v2(
        #     ux_neg,
        #     t,
        #     kind=kind,
        #     resolution=resolution,
        #     store_q=False,
        #     restrict=None
        # )
        # wz_pos = wass_v2(
        #     uz_pos,
        #     t,
        #     kind=kind,
        #     resolution=resolution,
        #     store_q=False,
        #     restrict=None
        # )
        # wz_neg = wass_v2(
        #     uz_neg,
        #     t,
        #     kind=kind,
        #     resolution=resolution,
        #     store_q=False,
        #     restrict=None
        # )
        evaluators.append([wx, wz])
    return evaluators
        

