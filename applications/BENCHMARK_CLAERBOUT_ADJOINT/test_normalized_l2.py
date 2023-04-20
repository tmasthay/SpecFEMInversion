from wass_v3 import *
import matplotlib.pyplot
import os
from typlotlib import *

def ricker(t0=0.0, sigma=1.0):
    def helper(t):
        u = (t-t0) / sigma
        return (1 - u**2) * np.exp( - u**2 / 2 )
    return helper

def get_waves(s, t, t0=0, sigma=1.0):
    u = []
    for (i,ss) in enumerate(s):
        v = ricker(t0+ss, sigma)
        u.append(v(t))
    ref = ricker(t0,sigma)
    return u, ref(t)

def get_dists(shifts, t, t0=0.0, sigma=1.0, renorm_func=square_normalize, sobolevs=[-1.0]):
    dt = t[1] - t[0]
    u, ref = get_waves(shifts,t,t0,sigma)
    ref_norm = renorm_func(ref, dt)
    w2_eval = wass_v3( ref_norm, t )
    w2_dists = []
    sobolev_dist = []
    l2_renorm_dist = []
    l2_dist = []
    other_sobolev = []
    for (i,e) in enumerate(u):
        u_norm = renorm_func(e, dt)
        res, g1, xi = sobolev_norm(u_norm-ref_norm, s=0.0, ot=t[0], nt=len(t), dt=dt)
        plt.plot(xi, g1, linestyle=':', color=rand_color())
        sobolev_dist.append(res)
        l2_renorm_dist.append(np.sum( (u_norm-ref_norm)**2 )  * dt)
        l2_dist.append( np.sum( (e - ref)**2 ) * dt )
        w2_dists.append( w2_eval(u_norm) )
    
    for (k,ee) in enumerate(sobolevs):
        other_sobolev.append([])
        for (i,e) in enumerate(u):
            res, g2, xi = sobolev_norm(u_norm-ref_norm, s=ee, ot=t[0], nt=len(t), dt=dt)
            plt.plot(xi, g2, linestyle=':', color=pre_colors[i+1], label='%d'%(i+2))
            plt.savefig('yup%d_%d.jpg'%(k,i))
            plt.close()
            other_sobolev[-1].append(res)
        os.system('convert -delay 10 -loop 0 yup%d_*.jpg yup%d.gif'%(k,k))
        os.system('open yup%d.gif'%k)
    return sobolev_dist, l2_renorm_dist, l2_dist, w2_dists, other_sobolev

def plot_dists(shifts, t, t0=0.0, sigma=1.0, renorm_func=square_normalize, sobolevs=[-1.0], use_random=True):
    markers = np.array(['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'])

    if( use_random ):
        precolors = [rand_color() for i in range(len(sobolevs) + 4)]
    else:
        precolors = pre_colors
    setup_gg_plot('black', 'black')
    sobolev_dist, l2_renorm_dist, l2_dist, w2_dists, other_sobolev = get_dists(shifts, t, t0, sigma, renorm_func, sobolevs)
    # plt.plot(shifts, sobolev_dist, color=precolors[0], linestyle='-', label='Renormalized Sobolev')
    # plt.plot(shifts, l2_renorm_dist, color=precolors[1], linestyle=':', label='Renormalized L2')
    # plt.plot(shifts, l2_dist, color=precolors[2], linestyle='-.', label='L2')
    plt.plot(shifts, w2_dists, color=precolors[3], label=r'$W_2^2$')
    for (i,e) in enumerate(sobolevs):
        plt.plot(
            shifts, 
            other_sobolev[i], 
            color=rand_color(), 
            linestyle=':', 
            label='%.2f'%e
        )
    plt.xlabel('Shift')
    plt.ylabel('Misfit')
    # plt.yscale('log')
    set_color_plot(use_legend=True)
    plt.savefig('zero_test.pdf')
    os.system('code zero_test.pdf')
                
t = np.linspace(-10,10,1000)
shifts = np.linspace(-5, 5, 100)
t0 = 0.0
sigma = 0.1
renorm_func = square_normalize
sobolevs = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0]
use_random = True

plot_dists(shifts, t, t0, sigma, renorm_func, sobolevs, use_random)
