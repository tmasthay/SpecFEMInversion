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
from helper_tyler import *
from concurrent.futures import ThreadPoolExecutor
import sys

def sobolev_norm(f, s=0, **kw):
    ot = kw['ot']
    dt = kw['dt']
    nt = kw['nt']
    xi = np.fft.fftfreq(nt, d=dt)
    f_hat = np.exp(-2j * np.pi * ot * xi) * np.fft.fft(f) * dt

    xi = np.fft.fftshift(xi)
    f_hat = np.fft.fftshift(f_hat)
    g = (1 + np.abs(xi)**2)**s * np.abs(f_hat)**2
    dxi = xi[1] - xi[0]
    res = np.trapz(g, dx=dxi)
    return res

def sobolev_multi(f, g, s=0, **kw):

    assert( len(f.shape) == 2 )

    renorm = kw.get('renorm', None)
    origin = kw['origin']
    delta = kw['delta']
    N = kw['N']

    assert( len(origin) == len(f.shape) )
    assert( len(delta) == len(f.shape) )
    assert( len(N) == len(f.shape) )
    if( renorm != None ):
        F = renorm(f)
        G = renorm(g)
    else:
        F = [f]
        G = [g]

    k = np.fft.fftshift(np.fft.fftfreq(N[0], d=delta[0]))
    omega = np.fft.fftshift(np.fft.fftfreq(N[1], d=delta[1])) 
    kernel = np.array(
        [
            [(1 + e_k**2 + e_omega**2)**s for e_omega in omega] \
                for e_k in k
        ]
    )
    total_sum = 0.0
    for FF,GG in zip(F,G):
        fourier_term = kernel * np.abs(np.fft.fftn(FF-GG))**2 * np.prod(delta)
        total_sum += np.trapz(
            np.trapz(fourier_term,dx=delta[0],axis=0),
            dx=delta[1],
            axis=0
        )
    return total_sum

def split_normalize(f, dx):
    f_abs = np.array(np.abs(f), dtype=np.float32)
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
    u = np.array(f**2, dtype=np.float32)
    c = np.trapz(u,dx=dx)
    return u if c == 0 else u / c

def shift_normalize(f, dx):
    c = np.min(f)
    if( c < 0 ):
        f = f + np.abs(c)
    c2 = np.trapz(f, dx=dx)
    if( c2 > 0 ):
        return f / c2
    else:
        return f

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
    x = np.array(x, dtype=np.float32)
    G = np.array(cumulative_trapezoid(g, dx=dx, initial=0.0), dtype=np.float32)
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
    x = np.asarray(x, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    q = smart_quantile_peval(g,x,tol,restrict,explicit_restrict)
    def helper(f, F=None):
        if( F == None ):
            F = cumulative_trapezoid(f, dx=x[1]-x[0], initial=0.0)
            F = np.asarray(F, dtype=np.float32)
        integrand = (q(F,tol) - x)**2*f
        return np.trapz(integrand, dx=x[1]-x[0])
    return helper
        
def create_evaluators(
        t,
        input_path='convex_reference',
        output_path='evaluators',
        **kw
):
    hf = helper()
    data_x = hf.read_SU_file('%s/Ux_file_single_d.su'%input_path)
    data_z = hf.read_SU_file('%s/Uz_file_single_d.su'%input_path)
    evaluators = []
    dt = t[1] - t[0]
    start_time = time.time()
    num_recs = data_x.shape[0]
    tau = kw.get('tau', 0.0)
    version = kw.get('version', 'split')
    make_plots = kw.get('make_plots', False)
    restrict = kw.get('restrict', None)
    explicit_restrict = kw.get('explicit_restrict', None)
    flush = kw.get('flush', False)
    s = kw.get('s', 0.0)
    ot = kw.get('ot', 0.0)
    dt = kw.get('dt', 0.1)
    nt = kw.get('nt', 101)
    renorm = kw.get('renorm', None)
    S = lambda g : lambda h : sobolev_norm(g-h, s=s, ot=ot, dt=dt, nt=nt)

    for i in range(num_recs):
        avg_time = (time.time() - start_time) / max(i,1)
        print('%d/%d ||| ELAPSED: %.2e ||| ETA: %.2e'%(
            i,
            data_x.shape[0],
            avg_time * max(i,1),
            avg_time * (num_recs - i)
            ),
            flush=flush
        )
        if( version.lower() == 'split' ):
            ux_pos, ux_neg = split_normalize(
                data_x[i], 
                dt 
            )
            uz_pos, uz_neg = split_normalize(
                data_z[i],
                dt
            )
            wx_pos = wass_v3(
                ux_pos,
                t,
                tol=tau,
                restrict=restrict,
                explicit_restrict=explicit_restrict
            )
            wx_neg = wass_v3(
                ux_neg,
                t,
                tol=tau,
                restrict=restrict,
                explicit_restrict=explicit_restrict
            )
            wz_pos = wass_v3(
                uz_pos,
                t,
                tol=tau,
                restrict=restrict,
                explicit_restrict=explicit_restrict
            )
            wz_neg = wass_v3(
                uz_neg,
                t,
                tol=tau,
                restrict=restrict,
                explicit_restrict=explicit_restrict
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
                    t, 
                    ux_pos, 
                    label='xpos', 
                    linestyle='-', 
                    color='green'
                )
                plt.plot(
                    t, 
                    ux_neg, 
                    label='xneg', 
                    linestyle='-.', 
                    color='red'
                )
                plt.legend()

                plt.subplot(1,3,3)
                plt.plot(
                    t,
                    cumulative_trapezoid(ux_pos, dx=dt, initial=0.0),
                    label='CDF xpos', 
                    linestyle='-', 
                    color='green'
                )
                plt.plot(
                    t,
                    cumulative_trapezoid(ux_neg, dx=dt, initial=0.0),
                    label='CDF xneg', 
                    linestyle='-.', 
                    color='red'
                )
                plt.savefig('split-%d.pdf'%i)
        elif( version.lower() == 'square' ):
            ux_norm = square_normalize(data_x[i], dt, clip_val=tau)
            uz_norm = square_normalize(data_z[i], dt, clip_val=tau)
            if( i < 5 ):
                plt.subplot(1,3,1)
                plt.plot(t, data_x[i], label='rawx', color='blue')
                plt.plot(t, data_z[i], label='rawz', color='red', linestyle='-.')
                plt.legend()

                plt.subplot(1,3,2)
                plt.plot(t, ux_norm, label='x', linestyle='-', color='green')
                plt.plot(t, uz_norm, label='z', linestyle='-.', color='red')
                plt.legend()

                plt.subplot(1,3,3)
                plt.plot(
                    t,
                    cumulative_trapezoid(ux_norm, dx=dt, initial=0.0),
                    label='Windowed CDF x', 
                    linestyle='-', 
                    color='green'
                )
                plt.plot(
                    t,
                    cumulative_trapezoid(uz_norm, dx=dt, initial=0.0),
                    label='Windowed CDF z', 
                    linestyle='-.', 
                    color='red'
                )
                plt.savefig('square-%d.pdf'%i)
            wx = wass_v3( 
                ux_norm,
                t,
                tol=tau,
                restrict=restrict,
                explicit_restrict=explicit_restrict
            )
            wz = wass_v3(
                uz_norm,
                t,
                tol=tau,
                restrict=restrict,
                explicit_restrict=explicit_restrict
            )
            evaluators.append([wx, wz])
        elif( version.lower() == 'sobolev' ):
            ux_cdf = cumulative_trapezoid(
                square_normalize(data_x[i], dx=dt),
                dx=dt,
                initial=0.0
            )
            uz_cdf = cumulative_trapezoid(
                square_normalize(data_z[i], dx=dt),
                dx=dt,
                initial=0.0
            )
            evaluators.append([S(ux_cdf), S(uz_cdf)])
        elif( version.lower() == 'sobolev_multi' ):
            pass
        else:
            raise ValueError('Mode "%s" not supported'%version)
    if( version.lower() == 'sobolev_multi' ):
        ox = kw.get('ox', 0.0)
        dx = kw.get('dx', 13.548167)
        kw.update(
            {
                'origin': np.array([ox, ot]), 
                'delta': np.array([dx,dt]),
                'N': data_x.shape
            }
        )
        S_multi = lambda f : lambda g : sobolev_multi(f,g,**kw)
        evaluators.append([S_multi(data_x), S_multi(data_z)])
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
    num_shifts = kw.get(
        'num_shifts',
        1 + np.max(np.array([e.split('_')[1:] for e in 
            ht.sco('find . -type d -name "convex_[0-9]*_[0-9]*"', True)],
            dtype=int)
        )
    )
    hf = helper()
    folders = [['ELASTIC/convex_%d_%d'%(i,j) for i in range(num_shifts)] \
        for j in range(num_shifts)]
    vals = np.zeros((num_shifts,num_shifts))
    
    num_recs = kw.get('num_recs', 501)
    version = kw.get('version', 'split')
    threaded = kw.get('threaded', False)
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
                    uxp_pdf, uxn_pdf = split_normalize(ux[k], dx=dt)
                    uzp_pdf, uzn_pdf = split_normalize(uz[k], dx=dt)
                    v1 = curr_xp(uxp_pdf)
                    v2 = curr_xn(uxn_pdf)
                    v3 = curr_zp(uzp_pdf)
                    v4 = curr_zn(uzn_pdf)
                    vals[i,j] += v1 + v2 + v3 + v4
                elif( version.lower() == 'square' ):
                    curr_x = evaluators[k][0]
                    curr_z = evaluators[k][1]
                    ux_pdf = square_normalize(ux[k], dx=dt)
                    uz_pdf = square_normalize(uz[k], dx=dt)
                    vals[i,j] += curr_x(ux_pdf) + curr_z(uz_pdf)
                elif( version.lower() == 'l2' ):
                    input_path = 'ELASTIC/convex_reference'
                    data_x = hf.read_SU_file('%s/Ux_file_single_d.su'%input_path)
                    data_z = hf.read_SU_file('%s/Uz_file_single_d.su'%input_path)
                    vals[i,j] = np.sum((data_x - ux)**2) + np.sum((data_z - uz)**2)
                elif( version.lower() == 'sobolev' ):
                    fx = lambda g : sobolev_norm(
                        data_x[i] - g,
                        s=s,
                        ot=ot,
                        dt=dt,
                        nt=nt
                    )


    return vals

def wass_landscape_threaded(evaluators, **kw):
    tau = kw.get('tau', 0.01)
    ot = kw.get('ot', 0.0)
    dt = kw.get('dt', 1.1e-3)
    path = kw.get('path', ht.sco('echo "$SPEC_APP"', True)[0])
    num_shifts = kw.get(
        'num_shifts',
        1 + np.max(np.array([e.split('_')[1:] for e in 
            ht.sco('find . -type d -name "convex_[0-9]*_[0-9]*"', True)],
            dtype=int)
        )
    )
    hf = helper()
    folders = [['%s/convex_%d_%d'%(path,i,j) for i in range(num_shifts)] \
        for j in range(num_shifts)]
    vals = np.zeros((num_shifts,num_shifts))
    
    # num_recs = kw.get('num_recs', 501)
    version = kw.get('version', 'split')
    version = version.replace('-', '_')
    # threaded = kw.get('threaded', False)

    global completed
    completed = 0

    def wrapper_split(index):
        global completed
        i = index // num_shifts
        j = np.mod(index, num_shifts)
        fx = '%s/Ux_file_single_d.su'%folders[i][j]
        fz = '%s/Uz_file_single_d.su'%folders[i][j]
        ux = hf.read_SU_file(fx)
        uz = hf.read_SU_file(fz)
        for k in range(ux.shape[0]):
            curr_xp = evaluators[k][0]
            curr_xn = evaluators[k][1]
            curr_zp = evaluators[k][2]
            curr_zn = evaluators[k][3]
            uxp_pdf, uxn_pdf = split_normalize(ux[k], dx=dt)
            uzp_pdf, uzn_pdf = split_normalize(uz[k], dx=dt)
            v1 = curr_xp(uxp_pdf)
            v2 = curr_xn(uxn_pdf)
            v3 = curr_zp(uzp_pdf)
            v4 = curr_zn(uzn_pdf)
            vals[i,j] += v1 + v2 + v3 + v4
        completed += 1
        return completed

    def wrapper_square(index):
        global completed
        i = index // num_shifts
        j = np.mod(index, num_shifts)
        fx = '%s/Ux_file_single_d.su'%folders[i][j]
        fz = '%s/Uz_file_single_d.su'%folders[i][j]
        ux = hf.read_SU_file(fx)
        uz = hf.read_SU_file(fz)
        for k in range(ux.shape[0]):
            curr_x = evaluators[k][0]
            curr_z = evaluators[k][1]
            ux_pdf = square_normalize(ux[k], dx=dt)
            uz_pdf = square_normalize(uz[k], dx=dt)
            vals[i,j] += curr_x(ux_pdf) + curr_z(uz_pdf)
        completed += 1
        return completed
    
    def wrapper_l2(index):
        global completed
        i = index // num_shifts
        j = np.mod(index, num_shifts)
        fx = '%s/Ux_file_single_d.su'%folders[i][j]
        fz = '%s/Uz_file_single_d.su'%folders[i][j]
        ux = hf.read_SU_file(fx)
        uz = hf.read_SU_file(fz)
        input_path = '%s/convex_reference'%path
        data_x = hf.read_SU_file('%s/Ux_file_single_d.su'%input_path)
        data_z = hf.read_SU_file('%s/Uz_file_single_d.su'%input_path)
        vals[i,j] = np.sum((data_x - ux)**2) + np.sum((data_z - uz)**2)
        completed += 1
        return completed
    
    def wrapper_sobolev(index):
        global completed
        i = index // num_shifts
        j = np.mod(index, num_shifts)
        fx = '%s/Ux_file_single_d.su'%folders[i][j]
        fz = '%s/Uz_file_single_d.su'%folders[i][j]
        ux = hf.read_SU_file(fx)
        uz = hf.read_SU_file(fz)
        for k in range(ux.shape[0]):
            curr_x = evaluators[k][0]
            curr_z = evaluators[k][1]
            ux_pdf = square_normalize(ux[k], dx=dt)
            uz_pdf = square_normalize(uz[k], dx=dt)
            ux_cdf = cumulative_trapezoid(ux_pdf, dx=dt, initial=0)
            uz_cdf = cumulative_trapezoid(uz_pdf, dx=dt, initial=0)
            vals[i,j] += curr_x(ux_cdf) + curr_z(uz_cdf)
        completed += 1
        return completed

    def wrapper_sobolev_multi(index):
        global completed
        i = index // num_shifts
        j = np.mod(index, num_shifts)
        fx = '%s/Ux_file_single_d.su'%folders[i][j]
        fz = '%s/Uz_file_single_d.su'%folders[i][j]
        ux = hf.read_SU_file(fx)
        uz = hf.read_SU_file(fz)
        vals[i,j] = evaluators[0][0](ux) + evaluators[0][1](uz)
        completed += 1
        return completed
    
    start_time = time.time()
    if( version.lower() == 'split' ):
        exec_function = wrapper_split
    elif( version.lower() == 'square' ):
        exec_function = wrapper_square
    elif( version.lower() == 'l2' ):
        exec_function = wrapper_l2
    elif( version.lower() == 'sobolev' ):
        exec_function = wrapper_sobolev
    elif( version.lower() == 'sobolev_multi' ):
        exec_function = wrapper_sobolev_multi
    else:
        raise ValueError('Version "%s" not supported'%version)

    with ThreadPoolExecutor() as executor:
        # Use a lambda function to pass additional arguments
        for res in executor.map(exec_function, range(num_shifts**2)):
            print(res)
    print('TOTAL TIME: %f'%(time.time() - start_time))
    return vals



    
        

