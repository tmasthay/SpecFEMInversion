from wasserstein import *
import numpy as np
from scipy.special import erf, erfinv
import matplotlib.pyplot as plt
import os

os.system('mkdir -p unit_test_plots/')
os.system('mkdir -p unit_test_plots/wasserstein/')

def get_gauss(mu, sig, a, b, nt):
    nt = int(nt)
    t = np.linspace(a, b, nt)
    dt = t[1] - t[0]
    u = np.exp(-(t-mu)**2 / (2 * sig**2))
    C = np.trapz(u) * dt
    return 1.0 / C * u, t, dt, a


def test_cdf():
    tol = 1e-4
    u, t, dt, ot = get_gauss(mu=0.0, sig=1.0, a=-10.0, b=10.0, nt=1e5)
    U = cumulative(u, False)
    ref = 0.5 * ( 1.0 + erf(t / np.sqrt(2)) )
    err = np.sqrt(dt) * np.linalg.norm(U - ref)
    assert err <= tol, "Incorrect CDF evalulation, (%.2e, %.2e)"%(err, tol)

def test_quantile():
    tol = 2e-3
    u,t,dt,ot = get_gauss(0.0, 1.0, -10.0, 10.0, 1e4)
    U = cumulative(u, True)
    p = np.linspace(0.01,0.99,int(1e6))
    Q = quantile(U, p, dt, ot=ot)
    ref = np.sqrt(2) * erfinv(2 * p - 1)
    err = max(Q - ref)

    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(t, u)
    plt.title('PDF')

    plt.subplot(2,2,2)
    plt.plot(t,U[1:])
    plt.title('CDF')

    plt.subplot(2,2,3)
    plt.plot(p,Q)
    plt.title('Quantile')

    plt.subplot(2,2,4)
    plt.plot(p, ref)
    plt.title('Reference Quantile')

    plt.savefig('unit_test_plots/wasserstein/test_quantile.pdf')
    assert err <= tol, "Incorrect quantile evalulation: (%.2e, %.2e)"%(err, tol)

def test_transport_distance():
    tol = 1e-5
    u, t, dt, ot = get_gauss(mu=0.0, sig=1.0, a=-10.0, b=10.0, nt=100)
    d, t2, dt2, ot2 = get_gauss(mu=0.0, sig=1.0, a=-10.0, b=10.0, nt=100)
    assert( (t == t2).all() )
    F = transport_distance(d=d,u=u,dt=dt,ot=ot)
    assert max(F[0]) <= tol, "Incorrect Wasserstein kernel: (%.2e, %.2e)"%(max(F), tol)

def test_wass_adjoint():
    tol = 1e-5
    restrict=0.1
    d, t, dt, ot = get_gauss(mu=0.0, sig=1.0, a=-10.0, b=10.0, nt=100)
    u, t2, dt2, ot2 = get_gauss(mu=0.0, sig=1.0, a=-10.0, b=10.0, nt=100)
    assert( (t == t2).all() )
    adj,Q,_,_ = wass_adjoint(d=d,u=u,dt=dt,ot=ot,restrict=restrict)
    err = np.linalg.norm(adj)
    assert err <= tol, "Incorrect Wasserstein adjoint: (%.2e, %.2e)"%(err, tol)

def test_wass_adjoint_and_eval():
    tol = 1e-5

    mu = 3.0
    restrict = 0.25
    nt = 1000

    u, t, dt, ot = get_gauss(mu=0.0, sig=1.0, a=-10.0, b=10.0, nt=nt)
    d, t2, dt2, ot2 = get_gauss(mu=mu, sig=1.0, a=-10.0, b=10.0, nt=nt)
    assert( (t == t2).all() )

    dist,adj,Q,D,U = wass_adjoint_and_eval(d=d,u=u,dt=dt,ot=ot,restrict=restrict)
    err = dist - mu**2

    i1,i2 = cut(len(t), restrict)
    print('%d,%d'%(i1,i2))
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(t, d, label='data')
    plt.plot(t, u, label='synthetic')
    plt.legend()

    plt.subplot(2,2,2)
    plt.plot(t[i1:i2], Q, label='Transport plan')
    plt.plot(t[i1:i2], Q**2*u[i1:i2], label='Transport normalized')
    #plt.plot(U, mu * np.ones(len(t)), label='reference')
    plt.legend()

    plt.subplot(2,2,3)
    plt.plot(t[i1:i2], adj, label='W2 adjoint')
    plt.plot(t[i1:i2], u[i1:i2] - d[i1:i2], label='L2 adjoint')
    plt.legend()

    plt.subplot(2,2,4)
    plt.plot(t[i1:i2], adj - Q**2, label='accumulator')
    plt.legend()

    plt.savefig('unit_test_plots/wasserstein/test_wass_adjoint_and_eval.pdf')

    assert err <= tol, "Incorrect Wasserstein adjoint+eval: (%.2e, %.2e)"%(err, tol)

def test_split_normalization():
    x = np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5])
    pos, neg = split_normalize(x)
    ref_pos = (1.0 / 15.0) * np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    ref_neg = (1.0 / 15.0) * np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert( len(ref_pos) == len(ref_neg) )
    valid = (pos == ref_pos).all() and (neg == ref_neg).all()

    plt.figure()
    N = len(x)
    plt.plot(range(N), x, linestyle='-', label='Orig')
    plt.plot(range(N), pos, linestyle='dashed', label='+')
    plt.plot(range(N), neg, linestyle='dashdot', label='-')
    plt.legend()
    plt.title('Splitting normalization')
    plt.savefig('unit_test_plots/wasserstein/split_normalize.pdf')
    assert valid, "Incorrect normalization"