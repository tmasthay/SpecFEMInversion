#!/usr/bin/env python

import numpy as np
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from subprocess import check_output as co
import sys
import re
from glob import *
import os
from helper_functions import *
from scipy.interpolate import RectBivariateSpline as RBS
from adj_seismogram import adj_seismogram

class ht:
    def sco(s, split_output=False):
        s = co(s,shell=True).decode('utf-8')
        if( split_output ):
            s = [e for e in s.split('\n') if e != '']
        return s
    
    def read_close(filename):
        with open(filename, 'r') as f:
            return f.read()
    
    def write_close(s, filename):
        with open(filename, 'w') as f:
            f.write(s)

    def append_close(s, filename):
        with open(filename, 'a') as f:
            f.write(s)

    def get_last(filename, conversion=float):
        return conversion(ht.sco('cat %s'%filename, True)[-1])

    def get_params(filename='DATA/Par_file_ref', type_map=dict()):
        try:
            d = dict()
            s = [re.sub('#.*$', '', e).strip() for e in open(filename, 'r').readlines()]
            s = [re.sub('[ ]+', ' ', e).strip() for e in s if e != '']
            for e in s:
                v = [ee.strip() for ee in e.split('=') if ee != '']
                if( len(v) == 1 ):
                    if( 'NO_FIELD' in d.keys() ): d['NO_FIELD'].append(v[0])
                    else: d['NO_FIELD'] = [v[0]]
                elif( len(v) == 2 ):
                    tm = str
                    if( v[0] in type_map.keys() ): tm = type_map[v[0]]
                    if( v[0] in d.keys() ): d[v[0]].append(tm(v[1]))
                    else: d[v[0]] = [tm(v[1])]
            return d
        except Exception as e:
            raise

    def par_pull(filename='DATA/Par_file_ref'):
        par_map = {'NSTEP': int,
            'DT': float,
            'NSOURCES': int,
            'NPROC': int,
            'SIMULATION_TYPE': int,
            'nrec': int,
            'xdeb': float,
            'xfin': float,
            'zdeb': float,
            'zfin': float
        }
        return ht.get_params(filename, par_map)

    def src_pull(filename='DATA/SOURCE'):
        tmp1 = lambda x: float(x.replace('d','e'))
        src_map = {'f0': float,
            'source_type': int,
            'time_function_type': int,
            'Mxx': tmp1,
            'Mzz': tmp1,
            'Mxz': tmp1,
            'factor': tmp1,
            'xs': float,
            'zs': float
        }
        return ht.get_params(filename, src_map)

    def create_ricker_time_derivative(base_dir='DATA', warn=False):
        if( base_dir[-1] == '/' ):
            base_dir = base_dir[:-1]

        par_fields = ht.par_pull(base_dir + '/Par_file_ref')
        source_fields = ht.src_pull(base_dir + '/SOURCE')

        nt = par_fields['NSTEP'][0]
        dt = par_fields['DT'][0]
        t = np.linspace(0.0, dt*(nt-1), nt)

        freq = source_fields['f0']
        for (i,f) in enumerate(freq):
            tmp1 = -6.0 * np.pi**2 * f * t**2
            tmp2 = 4.0 * np.pi**4 * f**3 * t**4
            tmp3 = np.exp(-np.pi**2 * f**2 * t**2)
            curr = (tmp1 + tmp2) * tmp3
            np.save('%s/ricker_time_deriv_%d.bin'%(base_dir, i), curr)

    def add_artificial_receivers(src, N=5, filename='DATA/Par_file', dz=1.0, dx=1.0, delete=False):
        if( not delete ):
            sp = lambda a,b: '%s%s= %s\n'%(a,28 * ' ',b)
            s = ''
            n = int(N/2)
            for i in range(-n,n+1):
                s += '# ARTIFICIAL RECEIVER GROUP %d\n'%(i+2)
                s += sp('nrec', str(N))
                s += sp('xdeb', '%.1f'%(src[0]-dx))
                s += sp('zdeb', '%.1f'%(src[1]+i*dz))
                s += sp('xfin', '%.1f'%(src[0]+dx))
                s += sp('zfin', '%.1f'%(src[1]+i*dz))
                s += 'record_at_surface_same_vertical = .false.\n\n'
            start_tag = '# ARTIFICIAL RECEIVERS START'
            f = open(filename, 'r')
            text = f.read()
            f.close()
            text = text.replace(start_tag, start_tag + '\n' + s)
            print(re.findall(r'%s'%start_tag, text))
            f = open(filename, 'w')
            f.write(text)
        else:
            s = ht.read_close(filename)
            start_tag = '# ARTIFICIAL RECEIVERS START'
            end_tag = '# ARTIFICIAL RECEIVERS END'
            inner_text = s.split(start_tag)[-1].split(end_tag)[0]
            s = s.replace(inner_text, '')
            ht.write_close(s, filename)
    
    def update_source(xs, zs, filename='DATA/SOURCE'):
        s = ht.read_close(filename)
        t = 28 * ' '
        s = re.sub('xs.*=.*', 'xs%s= %.8f'%(t,xs), s)
        s = re.sub('zs.*=.*', 'zs%s= %.8f'%(t,zs), s)
        ht.write_close(s,filename)

    def gd_adjoint(filename, n=3, dx=1.0, dz=1.0, kx=3, ky=3):
        hf = helper()
        f1 = hf.read_SU_file(filename[0])
        f2 = hf.read_SU_file(filename[1])
        N = n**2
        mid = int(n/2)
        v1 = f1[-N:].reshape((n,n,f1.shape[-1]))
        v2 = f2[-N:].reshape((n,n,f2.shape[-1]))
        nt = v1.shape[-1]
        X = np.array([i*dx for i in range(-mid,mid+1)])
        Z = np.array([i*dz for i in range(-mid,mid+1)])
        print(X)
        print(Z)
        print(v1.shape)
        print(v2.shape)
        splines1 = [RBS(X,Z,v1[:,:,i],kx=kx,ky=ky) for i in range(nt)]
        splines2 = [RBS(X,Z,v2[:,:,i],kx=kx,ky=ky) for i in range(nt)]
        mixed1 = np.array([u.partial_derivative(1,1)(0.0,0.0) for u in splines1])
        mixed2 = np.array([u.partial_derivative(1,1)(0.0,0.0) for u in splines2])
        laplace1 = np.array([u.partial_derivative(2,0)(0.0,0.0) for u in splines1])
        laplace2 = np.array([u.partial_derivative(0,2)(0.0,0.0) for u in splines2])

        grad_div1 = laplace1 + mixed2
        grad_div2 = laplace2 + mixed1

        grad_div1 = grad_div1.reshape((nt,))
        grad_div2 = grad_div2.reshape((nt,))

        # df1_dx_dx = (v1[0,1] - 2 * v1[1,1] + v1[2,1]) / dx**2
        # df2_dz_dz = (v2[1,0] - 2 * v2[1,1] + v2[1,2]) / dz**2 

        # df1_dx_dz = (v1[2,2] + v1[0,0] - v1[2,0] - v1[0,2]) / (4.0 * dx * dz)
        # df2_dx_dz = (v2[2,2] + v2[0,0] - v2[2,0] - v2[0,2]) / (4.0 * dx * dz)

        # grad_div1 = df1_dx_dx + df2_dx_dz
        # grad_div2 = df2_dz_dz + df1_dx_dz
        return np.array([grad_div1,grad_div2])
    
    def make_ricker(nt, dt, f, filename='OUTPUT_FILES/ricker.npy'):
        t = np.linspace(0,(nt-1)*dt, nt)
        tmp1 = (1.0 - 2 * np.pi * f**2 * t**2)
        tmp2 = np.exp(-np.pi**2 * f**2 * t**2)
        np.save(filename, tmp1*tmp2)

    def src_grad(filenames, sd, dt, n=3, dx=1.0, dz=1.0, 
        ricker_file='OUTPUT_FILES/ricker.npy',
        output_file='OUTPUT_FILES/src_grad.npy'):
        s = ht.gd_adjoint(filenames, n, dx, dz)
        M = np.array([ [sd['Mxx'][0], sd['Mxz'][0]], [sd['Mxz'][0], sd['Mzz'][0]] ])
        g = np.load(ricker_file)
        scaled = g * np.matmul(M,s)
        integral = dt * np.trapz(scaled, axis=1)
        np.save(output_file, integral)
        return integral

    def run_simulator(mode, **kw):
        output_name = kw.get('output_name', 'OUTPUT_FILES.syn.adjoint')
        s = ''
        if( mode.lower() == 'f' ):
            if( 'output_name' not in kw.keys() ):
                output_name = 'OUTPUT_FIILES.syn.forward'
            s = '''
                echo
                echo "running data forward simulation"
                echo
                ./change_simulation_type.pl -f

                # saving model files
                sed -i '' "s/^SAVE_MODEL .*=.*/SAVE_MODEL = gll/" DATA/Par_file

                ./run_this_example.sh > output.log

                # checks exit code
                if [[ $? -ne 0 ]]; then exit 1; fi

                echo
                mv -v output.log OUTPUT_FILES/

                # backup copy
                rm -rf %s
                cp -rp OUTPUT_FILES %s

                cp -v OUTPUT_FILES/*.su SEM/dat/
            '''%(output_name, output_name)
        elif( mode.lower() == 'a' ):
            kernel = kw.get('kernel', 'KERNELS')
            kernel = kernel.replace('/', '')
            s = '''
                ./change_simulation_type.pl -b

                ./run_this_example.sh noclean > output.log

                # checks exit code
                if [[ $? -ne 0 ]]; then exit 1; fi

                echo
                mv -v output.log OUTPUT_FILES/output.kernel.log

                # backup
                rm -rf %s
                cp -rp OUTPUT_FILES %s

                # kernels
                cp -vp OUTPUT_FILES/output.kernel.log %s/
                cp -vp OUTPUT_FILES/*_kernel.* %s/
            '''%(output_name, output_name, kernel, kernel)
        else:
            raise ValueError('Must be adjoint or forward mode')

    def backtrack_and_update(g, src_param, misfit_type='l2', 
        c_armijo=0.0001, alpha0=2.0, 
        src_file='DATA/SOURCE', data_dir='OUTPUT_FILES.dat.forward', 
        out_dir='OUTPUT_FILES.syn.backtrack', final_dir='OUTPUT_FILES.syn.forward'):

        xs_orig = src_param['xs'][0]
        zs_orig = src_param['zs'][0]
        phi_prime0 = np.linalg.norm(g)**2
        alpha = alpha0
        misfitx = [float(e) for e in \
            ht.read_close('misfitx.log').split('\n') if e != ''][-1]
        misfitz = [float(e) for e in \
            ht.read_close('misfitz.log').split('\n') if e != ''][-1]
        ref_misfit = misfitx + misfitz
        misfit = np.inf
        while( misfit > ref_misfit + c_armijo * alpha * phi_prime0 ):
            alpha = alpha / 2.0
            print('Backtrack for alpha=%.2e'%alpha)
            xs = xs_orig + alpha * g[0]
            zs = zs_orig + alpha * g[1]
            ht.update_source(xs,zs,src_file)
            ht.run_simulator('forward', output_name=out_dir)
            adj_seismogram('%s/Ux_file_single_d.su'%out_dir, 
                '%s/Ux_file_single_d.su'%data_dir,
                misfit_type,
                '%s/misfitx.log'%out_dir)
            adj_seismogram('%s/Uz_file_single_d.su'%out_dir, 
                '%s/Uz_file_single_d.su'%data_dir,
                misfit_type,
                '%s/misfitz.log'%out_dir)
            ref_misfit = ht.get_last('%s/misfitx.log'%out_dir) \
                + ht.get_last('%s/misfitz.log'%out_dir)
        print('Successfully backtracked! alpha=%.2e'%(alpha))
        print('Moving backtrack directory "%s" to "%s"'%(out_dir, final_dir))
        os.system('rm -rf %s'%final_dir)
        os.system('mv %s %s'%(out_dir, final_dir))
        return xs,zs,alpha
            

if( __name__ == "__main__" ):
    mode = int(sys.argv[1])
    if( mode == 1 ):
        ht.create_ricker_time_derivative('ELASTIC/DATA')
        par_fields = ht.par_pull('ELASTIC/DATA/Par_file_ref')
        nt = par_fields['NSTEP'][0]
        dt = par_fields['DT'][0]
        t = np.linspace(0.0, dt * (nt-1), nt)

        source_params = ht.src_pull('ELASTIC/DATA/SOURCE')
        freq = source_params['f0']

        v = [np.load(e) for e in glob('ELASTIC/DATA/ricker_time_deriv_[0-9]*.bin*')]

        for (i,e) in enumerate(v):
            plt.plot(t, e, label='%.1f'%freq[i])
            plt.savefig('%d.pdf'%i)

        plt.legend()
        plt.title('Frequency comparison')
        plt.savefig('freq.pdf')
    elif( mode == 2 ):
        delete = False
        if( len(sys.argv) == 3 ):
            delete = sys.argv[2].lower()[0] == 't'
        v = ht.src_pull()
        src = [v['xs'][0], v['zs'][0]]
        ht.add_artificial_receivers(src, delete=delete)
    elif( mode == 4 ):
        pp = ht.par_pull()
        sp = ht.src_pull()
        freq = sp['f0'][0]
        xs = sp['xs'][0]
        zs = sp['zs'][0]
        nt = pp['NSTEP'][0]
        dt = pp['DT'][0]
        ht.make_ricker(nt,dt,freq)
        bd = 'OUTPUT_FILES.syn.adjoint'
        filenames = ['%s/Ux_file_single_d.su'%bd, '%s/Uz_file_single_d.su'%bd]
        s = ht.src_grad(filenames, sp, dt)
        ht.append_close(str(s), 'OUTPUT_FILES/grad.log')
    elif( mode == 5 ):
        xs = float(sys.argv[2])
        zs = float(sys.argv[3])
        ht.update_source(xs,zs)
    elif( mode == 6 ):
        perturb_percent = float(sys.argv[2])
        sp = ht.src_pull()
        xs = sp['xs'][0]
        zs = sp['zs'][0]
        pxs = 2.0 * (np.random.random() - 0.5) * perturb_percent / 100.0
        pzs = 2.0 * (np.random.random() - 0.5) * perturb_percent / 100.0
        u = xs
        v = zs
        xs = (1.0 + pxs) * xs
        zs = (1.0 + pzs) * zs
        print('(%f, %f) -> (%f, %f)'%(u,v,xs,zs), file=sys.stderr)
        ht.update_source(xs,zs)
        ht.add_artificial_receivers([xs,zs], filename='DATA/Par_file', delete=True)
        ht.add_artificial_receivers([xs,zs], filename='DATA/Par_file')




