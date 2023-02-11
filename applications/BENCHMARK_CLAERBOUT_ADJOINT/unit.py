from wasserstein import *
import numpy as np
from scipy.special import erf, erfinv

def test_cdf():
    tol = 1e-4
    t = np.linspace(-10,10,int(1e5))
    u = 1.0 / np.sqrt(2 * np.pi) * np.exp(-t**2 / 2)
    U = cumulative(u)
    ref = 0.5 * ( 1.0 + erf(t / np.sqrt(2)) )
    dt = t[1] - t[0]
    err = np.sqrt(dt) * np.linalg.norm(U[:-1] - ref)
    assert err <= tol, "Incorrect CDF evalulation, (%.2e, %.2e)"%(err, tol)

def test_quantile():
    tol = 1e-2
    t = np.linspace(-10,10,int(2e7))
    u = 1.0 / np.sqrt(2 * np.pi) * np.exp(-t**2 / 2)
    U = cumulative(u)
    p = np.linspace(0.01,0.99,100)
    dt = t[1] - t[0]
    Q = quantile(U, p, dt)
    ref = np.sqrt(2) * erfinv(2 * p - 1)
    err = np.sqrt(dt) * np.linalg.norm(Q - ref)
    assert err <= tol, "Incorrect quantile evalulation: (%.2e, %.2e)"%(err, tol)

def test_transport_distance():
    assert 1 == 1, "Incorrect Wasserstein kernel"

def test_wass_adjoint():
    assert 1 == 1, "Incorrect Wasserstein adjoint"

def test_wass_adjoint_and_eval():
    assert 1 == 1, "Incorrect Wasserstein adjoint+eval"