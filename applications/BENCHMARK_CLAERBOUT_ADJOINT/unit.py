from wasserstein import *
import numpy as np
from scipy.special import erf, erfinv
import matplotlib.pyplot as plt
import os
from helper_tyler import *
from helper_functions import *
from scipy.interpolate import RectBivariateSpline as RBS

os.system('mkdir -p unit_test_plots/')
os.system('mkdir -p unit_test_plots/wasserstein/')

### BEGIN UNIT TESTS FOR wasserstein.py ###
test_wasserstein = True
if( test_wasserstein ):
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
### END TESTS FOR wasserstein.py ###

test_helper_tyler = True

if( test_helper_tyler ):
    unit_test_dir = 'unit_tests'
    curr_test_env = 'helper_tyler'
    curr_dir = '%s/%s'%(unit_test_dir, curr_test_env)
    os.system('mkdir -p %s/'%unit_test_dir)
    os.system('mkdir -p %s'%curr_dir)

    ### BEGIN TESTS FOR helper_tyler.py ###
    def test_gd_adjoint():
        hf = helper()

        filenames = ['%s/xcomp.su'%curr_dir, '%s/zcomp.su'%curr_dir]
        xs = 1.0
        zs = 1.0
        dx = 1e-2
        dz = 1e-2
        n = 7
        mid = int(n/2)
        kx = 3
        kz = 3

        f1 = lambda x,z,t: np.sin(t*x*z)
        f2 = lambda x,z,t: np.exp(t*x*z)

        t = np.linspace(0, 1, 100)
        domain = np.array([[(xs+j*dx, zs+i*dz) for i in range(-mid,mid+1)] \
            for j in range(-mid,mid+1)])
        domain = domain.reshape((n**2,2))
        vals1 = np.array([ [f1(xx,zz,tt) for tt in t] for (xx,zz) in domain])
        vals2 = np.array([ [f2(xx,zz,tt) for tt in t] for (xx,zz) in domain])
        hf.write_SU_file(vals1, filenames[0])
        hf.write_SU_file(vals2, filenames[1])

        tmp1 = hf.read_SU_file(filenames[0])
        tmp2 = hf.read_SU_file(filenames[1])

        v1 = vals1.reshape((n,n,vals1.shape[-1]))
        v2 = vals2.reshape((n,n,vals2.shape[-1]))

        N = n**2
        # tmp1 = tmp1[-N:].reshape((n,n,tmp1.shape[-1]))
        # tmp2 = tmp2[-N:].reshape((n,n,tmp2.shape[-1]))
        tmp1 = tmp1[-N:].reshape((n,n,tmp1.shape[-1]))
        tmp2 = tmp2[-N:].reshape((n,n,tmp2.shape[-1]))

        output = ht.gd_adjoint(filenames, n, dx, dz, kx, kz)

        nt = v1.shape[-1]
        X = np.array([i*dx for i in range(-mid,mid+1)])
        Z = np.array([i*dz for i in range(-mid,mid+1)])
        splines1 = [RBS(X,Z,tmp1[:,:,i],kx=kx,ky=kz) for i in range(nt)]
        splines2 = [RBS(X,Z,tmp2[:,:,i],kx=kx,ky=kz) for i in range(nt)]
        mixed1 = np.array([u.partial_derivative(1,1)(0.0,0.0) for u in splines1])
        mixed2 = np.array([u.partial_derivative(1,1)(0.0,0.0) for u in splines2])
        laplace1 = np.array([u.partial_derivative(2,0)(0.0,0.0) for u in splines1])
        laplace2 = np.array([u.partial_derivative(0,2)(0.0,0.0) for u in splines2])

        grad_div1 = laplace1 + mixed2
        grad_div2 = laplace2 + mixed1

        grad_div1 = grad_div1.reshape((nt,))
        grad_div2 = grad_div2.reshape((nt,))
        direct = np.array([grad_div1, grad_div2])

        f3 = lambda x,z,t: -(t*z)**2 * np.sin(t*x*z) + t**2 * x * z * np.exp(t*x*z) + t * np.exp(t*x*z)
        f4 = lambda x,z,t: t*np.cos(t*x*z) - t**2 * x * z * np.sin(t*x*z) + (t*x)**2 * np.exp(t*x*z)
        ref = np.transpose([ (f3(xs,zs,tt), f4(xs,zs,tt)) for tt in t])
        do_plotting = True
        if( do_plotting ):
            plt.subplot(2,1,1)
            plt.plot(t, output[0], label='X computed')
            plt.plot(t, ref[0], linestyle='dashdot', label='X ref')
            plt.legend()

            plt.subplot(2,1,2)
            plt.plot(t, output[1], label='Z computed')
            plt.plot(t, ref[1], linestyle='dashdot', label='Z ref')
            plt.legend()
            plt.savefig('%s/gd_adjoint.pdf'%curr_dir)

        err = np.max(abs(output - ref))
        tol = 1e-2
        assert err <= tol, '(computed, target) = (%.2e, %.2e)'%(err, tol)