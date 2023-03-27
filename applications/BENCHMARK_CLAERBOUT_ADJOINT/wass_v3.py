import numpy as np
from itertools import accumulate
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.stats.sampling import NumericalInverseHermite
from helper_functions import helper
import time
import pickle
import matplotlib.pyplot as plt
from smart_quantile_cython import *

def split_normalize(f, dx):
    f_abs = np.abs(f)
    pos = 0.5 * (f_abs + f)
    neg = pos - f
    c_pos = np.trapz(pos, dx=dx)
    c_neg = np.trapz(neg, dx=dx)
    if( c_pos > 0 ):
        pos /= c_pos
    if( c_neg > 0 ):
        neg /= c_neg
    return pos,neg
    
def square_normalize(f, dx, clip_val=None):
    u = f**2
    c = np.trapz(u,dx=dx)
    return u if c == 0 else u / c

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
    
def smart_quantile_peval(
        g,
        x,
        tol=0.0, 
        restrict=None,
        explicit_restrict=None
    ):
    dx = x[1] - x[0]
    G = cumulative_trapezoid(g, dx=dx, initial=0.0)
    def helper(p, tau=tol):
        return smart_quantile(x, g, G, p, tol)
    return helper

def wass_v3(
        g, 
        x,
        tol=0.0,
        restrict=None,
        explicit_restrict=None
):
    q = smart_quantile_peval(g,x,tol,restrict,explicit_restrict)
    def helper(f, F=None):
        if( F == None ):
            F = cumulative_trapezoid(f, dx=x[1]-x[0], initial=0.0)
        integrand = (q(F,tol) - x)**2*f
        return np.trapz(integrand, dx=x[1]-x[0])
        
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
    tau = kw.get('tau', 0.01)
    version = kw.get('version', 'split')
    make_plots = kw.get('make_plots', False)
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
        if( version.lower() == 'split' ):
            ux_pos, ux_neg, ixp, ixn, cxp, cxn = split_normalize(
                data_x[i], 
                dt, 
                clip_val=tau
            )
            uz_pos, uz_neg, izp, izn, czp, czn = split_normalize(
                data_z[i],
                dt, 
                clip_val=tau
            )
            wx_pos = wass_v2(
                ux_pos,
                t,
                kind=kind,
                resolution=resolution,
                store_q=False,
                restrict=None
            )
            wx_neg = wass_v2(
                ux_neg,
                t,
                kind=kind,
                resolution=resolution,
                store_q=False,
                restrict=None
            )
            wz_pos = wass_v2(
                uz_pos,
                t,
                kind=kind,
                resolution=resolution,
                store_q=False,
                restrict=None
            )
            wz_neg = wass_v2(
                uz_neg,
                t,
                kind=kind,
                resolution=resolution,
                store_q=False,
                restrict=None
            )
            evaluators.append([wx_pos, wx_neg, wz_pos, wz_neg])
            if( i < 5 and make_plots):
                plt.clf()
                plt.subplot(1,3,1)
                plt.plot(t, data_x[i], label='rawx', color='blue')
                plt.plot(
                    t, 
                    data_z[i], 
                    label='rawz', 
                    color='red', 
                    linestyle='-.'
                )
                plt.legend()

                plt.subplot(1,3,2)
                plt.plot(
                    t[ixp], 
                    ux_pos[ixp], 
                    label='xpos', 
                    linestyle='-', 
                    color='green'
                )
                plt.plot(
                    t[ixn], 
                    ux_neg[ixn], 
                    label='xneg', 
                    linestyle='-.', 
                    color='red'
                )
                plt.legend()

                plt.subplot(1,3,3)
                plt.plot(
                    t[ixp],
                    cumulative_trapezoid(ux_pos[ixp], dx=dt, initial=0.0),
                    label='Windowed CDF xpos', 
                    linestyle='-', 
                    color='green'
                )
                plt.plot(
                    t[ixn],
                    cumulative_trapezoid(ux_neg[ixn], dx=dt, initial=0.0),
                    label='Windowed CDF xneg', 
                    linestyle='-.', 
                    color='red'
                )
                plt.savefig('split-%d.pdf'%i)
        elif( version.lower() == 'square' ):
            ux_norm, ix, cx = square_normalize(data_x[i], dt, clip_val=tau)
            uz_norm, iz, cz = square_normalize(data_z[i], dt, clip_val=tau)
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
                plt.savefig('square-%d.pdf'%i)
            wx = wass_v2(
                ux_norm[ix] / cx,
                t[ix],
                kind=kind,
                resolution=resolution,
                store_q=False,
                restrict=None,
                explicit_restrict=None
            )
            wz = wass_v2(
                uz_norm[iz] / cz,
                t[iz],
                kind=kind,
                resolution=resolution,
                store_q=False,
                restrict=None,
                explicit_restrict=None
            )
            evaluators.append([wx, wz])
    return evaluators

def get_info(f, dx, tau=None, version='split'):
    if( version.lower() == 'split' ):
        fp, fn, ixp, ixn, cpos, cneg = split_normalize(f, dx=dx, clip_val=tau)
        Fp = cumulative_trapezoid(fp, dx=dx, initial=0.0)
        Fn = cumulative_trapezoid(fn, dx=dx, initial=0.0)
        return fp, fn, Fp, Fn, ixp, ixn, cpos, cneg
    else:
        f_norm, ix, c = square_normalize(f, dx, clip_val=tau)
        F = cumulative_trapezoid(f_norm, dx=dx, initial=0.0)
        return f_norm, F, ix, c

def wass_landscape(evaluators, **kw):
    tau = kw.get('tau', 0.01)
    dt = kw.get('dt', 0.00140)
    num_shifts = kw.get('num_shifts', 100)
    hf = helper()
    folders = [['ELASTIC/convex_%d_%d'%(i,j) for i in range(num_shifts)] \
        for j in range(num_shifts)]
    vals = np.zeros((num_shifts,num_shifts))
    
    num_recs = kw.get('num_recs', 501)
    version = kw.get('version', 'split')
    start_time = time.time()
    for i in range(num_shifts):
        for j in range(num_shifts):
            avg_time = (time.time() - start_time) / max(100*i+j,1)
            print('(%d,%d)/(%d,%d) *** ELAPSED: %.2e *** ETA: %.2e'%(
                i,
                j,
                100,
                100,
                avg_time * max(i*100+j,1),
                avg_time * (num_shifts**2 - (i*100+j))
                ),
                flush=True
            )
            fx = '%s/Ux_file_single_d.su'%folders[i][j]
            fz = '%s/Uz_file_single_d.su'%folders[i][j]
            ux = hf.read_SU_file(fx)
            uz = hf.read_SU_file(fz)
            s = 0.0
            for k in range(ux.shape[0]):
                if( version.lower() == 'split' ):
                    curr_xp = evaluators[k][0]
                    curr_xn = evaluators[k][1]
                    curr_zp = evaluators[k][2]
                    curr_zn = evaluators[k][3]
                    uxp_pdf, uxp_cdf, uxn_pdf, uxn_cdf, ixp, ixn, cxp, cxn = \
                    get_info(
                        ux[k], 
                        dx=dt, 
                        tau=tau,
                        version=version
                    )
                    uzp_pdf, uzp_cdf, uzn_pdf, uzn_cdf, izp, izn, czp, czn = \
                    
                    \get_info(
                        uz[k], 
                        dx=dt, 
                        tau=tau,
                        version=version
                    )
                    v1 = curr_xp(uxp_pdf, uxp_cdf)
                    v2 = curr_xn(uxn_pdf, uxn_cdf)
                    v3 = curr_zp(uzp_pdf, uzp_cdf)
                    v4 = curr_zn(uzn_pdf, uzn_cdf)
                    vals[i,j] += v1 + v2 + v3 + v4
                else:
                    curr_x = evaluators[k][0]
                    curr_z = evaluators[k][1]
                    ux_pdf, ux_cdf, ix, cx = get_info(
                        ux[k], 
                        dx=dt, 
                        tau=tau,
                        version=version
                    )
                    uz_pdf, uz_cdf, iz, cz = get_info(
                        uz[k], 
                        dx=dt, 
                        tau=tau,
                        version=version
                    )
                    vals[i,j] += curr_x(ux_pdf, ux_cdf) + curr_z(uz_pdf, uz_cdf)
    return vals


    
        

