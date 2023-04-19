from wass_v3 import *
import matplotlib.pyplot
import os

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

def get_dists(shifts, t, t0=0.0, sigma=1.0, renorm_func=square_normalize):
    dt = t[1] - t[0]
    u, ref = get_waves(shifts,t,t0,sigma)
    ref_norm = renorm_func(ref, dt)
    sobolev_dist = []
    l2_renorm_dist = []
    l2_dist = []
    for (i,e) in enumerate(u):
        u_norm = renorm_func(e, dt)
        sobolev_dist.append(sobolev_norm(u_norm-ref_norm, s=0.0, ot=t[0], nt=len(t), dt=dt))
        l2_renorm_dist.append(np.sum( (u_norm-ref_norm)**2 )  * dt)
        l2_dist.append( np.sum( (e - ref)**2 ) * dt )
    return sobolev_dist, l2_renorm_dist, l2_dist

def plot_dists(shifts, t, t0=0.0, sigma=1.0, renorm_func=square_normalize):
    sobolev_dist, l2_renorm_dist, l2_dist = get_dists(shifts, t, t0, sigma, renorm_func)
    plt.plot(shifts, sobolev_dist, color='red', linestyle='-', label='Renormalized Sobolev')
    plt.plot(shifts, l2_renorm_dist, color='blue', linestyle=':', label='Renormalized L2')
    plt.plot(shifts, l2_dist, color='purple', linestyle='-.', label='L2')
    plt.xlabel('Shift')
    plt.ylabel('Misfit')
    plt.yscale('log')
    plt.legend()
    plt.savefig('zero_test.pdf')
    os.system('code zero_test.pdf')
        
        
t = np.linspace(-10,10,1000)
shifts = np.linspace(-5, 5, 100)
t0 = 0.0
sigma = 1.0
renorm_func = shift_normalize

plot_dists(shifts, t, t0, sigma, renorm_func)
