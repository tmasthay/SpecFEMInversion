from wasserstein import *
import numpy as np
from scipy.special import erf, erfinv
import matplotlib.pyplot as plt
import os

os.system('mkdir -p unit_test_plots/')
os.system('mkdir -p unit_test_plots/wasserstein/')

def test_cdf():
    tol = 1e-4
    t = np.linspace(-10,10,int(1e5))
    u = 1.0 / np.sqrt(2 * np.pi) * np.exp(-t**2 / 2)
    U = cumulative(u, False)
    ref = 0.5 * ( 1.0 + erf(t / np.sqrt(2)) )
    dt = t[1] - t[0]
    err = np.sqrt(dt) * np.linalg.norm(U - ref)
    assert err <= tol, "Incorrect CDF evalulation, (%.2e, %.2e)"%(err, tol)

def test_quantile():
    tol = 1e-2
    t = np.linspace(-10,10,int(2e7))
    u = 1.0 / np.sqrt(2 * np.pi) * np.exp(-t**2 / 2)
    U = cumulative(u, True)
    p = np.linspace(0.01,0.99,100)
    dt = t[1] - t[0]
    Q = quantile(U, p, dt, ot=-10.0)
    ref = np.sqrt(2) * erfinv(2 * p - 1)
    err = np.sqrt(dt) * np.linalg.norm(Q - ref)

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
    d = np.random.random(100)
    u = d
    F = transport_distance(d,u,1.0,0.0)
    assert max(F) <= tol, "Incorrect Wasserstein kernel: (%.2e, %.2e)"%(max(F), tol)

def test_wass_adjoint():
    tol = 1e-5
    d = np.random.random(100)
    u = d
    adj,Q = wass_adjoint(d=d,u=u,dt=1.0,ot=0.0)
    err = np.linalg.norm(adj)
    assert err <= tol, "Incorrect Wasserstein adjoint: (%.2e, %.2e)"%(err, tol)

def test_wass_adjoint_and_eval():
    tol = 1e-5

    mu = 0.0
    a = 10.0
    nt = int(1e5)
    t = np.linspace(-a,a,nt)
    dt = t[1] - t[0]

    d = 1.0 / np.sqrt(2 * np.pi) * np.exp(-t**2 / 2)
    u = 1.0 / np.sqrt(2 * np.pi) * np.exp(-(t-mu)**2 / 2)
    dist,adj,Q = wass_adjoint_and_eval(d=d,u=u,dt=dt,ot=-a)
    err = dist - mu**2

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(t, d, label='data')
    plt.plot(t, u, label='synthetic')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(t, Q, label='transport distance')
    plt.plot(t, mu * np.ones(len(t)), label='reference')
    plt.legend()

    plt.savefig('unit_test_plots/wasserstein/test_wass_adjoint_and_eval.pdf')

    assert err <= tol, "Incorrect Wasserstein adjoint+eval: (%.2e, %.2e)"%(err, tol)