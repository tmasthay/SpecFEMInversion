from wasserstein import *
import numpy as np
from scipy.special import erf, erfinv
import matplotlib.pyplot as plt
import os
from helper_tyler import *
from helper_functions import *

os.system('mkdir -p unit_test_plots/')
os.system('mkdir -p unit_test_plots/wasserstein/')

### BEGIN UNIT TESTS FOR wasserstein.py ###
test_wasserstein = False
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
        dx = 1e-1
        dz = 1e-1
        n = 3

        domain = [ [(xs+j*dx, zs+i*dz) for i in range(-1,2)] for j in range(-1,2) ]
        print(str(domain))
        domain = np.array(domain).reshape((9,2))
        print(domain)
        # domain = domain.reshape((3,3,2))
        # print(domain)

        f1 = lambda x,z,t: np.sin(t*x*z)
        f2 = lambda x,z,t: np.exp(t*x*z**2)

        t = np.linspace(0, 1, 100)
        vals1 = np.array([ [f1(xx,zz,tt) for tt in t] for (xx,zz) in domain])
        vals2 = np.array([ [f2(xx,zz,tt) for tt in t] for (xx,zz) in domain])
        tmp1 = vals1.copy().reshape((n,n,vals1.shape[-1]))
        tmp2 = vals2.copy().reshape((n,n,vals2.shape[-1]))
        hf.write_SU_file(vals1, filenames[0])
        hf.write_SU_file(vals2, filenames[1])

        vals1 = hf.read_SU_file(filenames[0])
        vals2 = hf.read_SU_file(filenames[1])

        v1 = vals1.reshape((n,n,vals1.shape[-1]))
        v2 = vals2.reshape((n,n,vals2.shape[-1]))

        # v1 = np.transpose(v1, axes=(1,0,2))
        # v2 = np.transpose(v2, axes=(1,0,2))

        df1_dx_dx = (v1[0,1] - 2 * v1[1,1] + v1[2,1]) / dx**2
        df2_dz_dz = (v2[1,0] - 2 * v2[1,1] + v2[1,2]) / dz**2 

        df1_dx_dz = (v1[2,2] + v1[0,0] - v1[2,0] - v1[0,2]) / (4.0 * dx * dz)
        df2_dx_dz = (v2[2,2] + v2[0,0] - v2[2,0] - v2[0,2]) / (4.0 * dx * dz)

        grad_div1 = df1_dx_dx + df2_dx_dz
        grad_div2 = df2_dz_dz + df1_dx_dz

        u1 = v1[:,:,0]
        u2 = v2[:,:,0]
        w1 = tmp1[:,:,0]
        w2 = tmp2[:,:,0]
        print(u1)
        print(w1)
        print(u2)
        print(w2)
        # print(df1_dx_dx[0])
        # print(df2_dz_dz[0])
        # print(df1_dx_dz[0])
        # print(df2_dx_dz[0])
        # print('(%f,%f,%f) --> %f'%(v1[0,1,0], v1[1,1,0], v1[2,1,0], v1[0,1,0] - 2 * v1[1,1,0] + v1[2,1,0]))
        # print('v1 = %s'%(str(v1)))
        # print('v2 = %s'%(str(v2)))


        output = ht.gd_adjoint(filenames, n=n, dx=dx, dz=dz, use_double=False)
        # output = np.array([grad_div1, grad_div2])

        f3 = lambda x,z,t: p*(p-1)*x**(p-2)
        f4 = lambda x,z,t: q*(q-1)*z**(p-2)
        ref = np.transpose([ (f3(xs,zs,tt), f4(xs,zs,tt)) for tt in t])
        do_plotting = True
        if( do_plotting ):
            plt.subplot(2,1,1)
            plt.plot(t, output[0], label='X computed')
            plt.plot(t, ref[0], linestyle='dashdot', label='X ref')
            plt.plot(t, grad_div1, linestyle=(0,(1,10)), label='ref_diff')
            plt.legend()

            plt.subplot(2,1,2)
            plt.plot(t, output[1], label='Z computed')
            plt.plot(t, ref[1], linestyle='dashdot', label='Z ref')
            plt.plot(t, grad_div2, linestyle=(0,(1,10)), label='ref_diff')
            plt.legend()
            plt.savefig('%s/gd_adjoint.pdf'%curr_dir)

        err = np.max(abs(output - ref))
        tol = 1e-4
        assert err <= tol, '(computed, target) = (%.2e, %.2e)'%(err, tol)