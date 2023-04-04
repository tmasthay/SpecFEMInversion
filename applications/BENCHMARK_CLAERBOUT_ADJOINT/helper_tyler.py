#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from subprocess import check_output as co
import sys
import re
from glob import *
import os
from helper_functions import *
from scipy.interpolate import RectBivariateSpline as RBS
from adj_seismogram import eval_misfit, adj_seismogram
from itertools import product
import argparse
import traceback
import imageio
import time

class ht:
    def sco(
            s, 
            split_output=False
            ):
        s = co(s,shell=True).decode('utf-8')
        if( split_output ):
            s = [e for e in s.split('\n') if e != '']
        return s
        
    def read_close(
            filename
            ):
        with open(filename, 'r') as f:
            return f.read()
    
    def write_close(
            s, 
            filename
            ):
        with open(filename, 'w') as f:
            f.write(s)

    def append_close(
            s, 
            filename
            ):
        with open(filename, 'a') as f:
            f.write(s)

    def get_last(
            filename, 
            conversion=float
            ):
        return conversion(ht.sco('cat %s'%filename, True)[-1])

    def get_params(
            filename='DATA/Par_file_ref', 
            type_map=dict()
            ):
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
                    if( tm == float ): v[1] = v[1].replace('d', 'e')
                    if( v[0] in d.keys() ): d[v[0]].append(tm(v[1]))
                    else: d[v[0]] = [tm(v[1])]
            return d
        except Exception as e:
            raise

    def par_pull(
            filename='DATA/Par_file_ref'
            ):
        par_map = {
            'NSTEP': int,
            'DT': float,
            'NSOURCES': int,
            'NPROC': int,
            'SIMULATION_TYPE': int,
            'nrec': int,
            'xdeb': float,
            'xfin': float,
            'zdeb': float,
            'zfin': float,
            'nreceiversets': int
        }
        return ht.get_params(filename, par_map)

    def src_pull(
            filename='DATA/SOURCE'
            ):
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

    def create_ricker_time_derivative(
            base_dir='DATA', 
            warn=False
            ):
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

    def update_field(
            field_name, 
            value, 
            filename
            ):
        s = ht.read_close(filename)
        t = 28 * ' '
        if( type(value) == int ):
            s = re.sub('%s.*=.*'%field_name, '%s%s= %d'%(field_name, t, value), s)
        elif( type(value) == float ):
            s = re.sub('%s.*=.*'%field_name, '%s%s= %.8f'%(field_name, t, value), s)
        else:
            s = re.sub('%s.*=.*'%field_name, '%s%s= %s'%(field_name, t, value), s)
        ht.write_close(s, filename)

    def update_source(
            xs, 
            zs, 
            filename='DATA/SOURCE'
            ):
        ht.update_field('xs', xs, filename)
        ht.update_field('zs', zs, filename)

    def add_artificial_receivers(
            src, 
            og_recs, 
            N=5, 
            filename='DATA/Par_file', 
            dz=1.0, 
            dx=1.0
            ):
        ht.update_field('nreceiversets', og_recs + N, filename)

        s = ht.read_close(filename)
        start_tag = '# ARTIFICIAL RECEIVERS START'
        end_tag = '# ARTIFICIAL RECEIVERS END'
        inner_text = s.split(start_tag)[-1].split(end_tag)[0]
        s = s.replace(inner_text, '')
        ht.write_close(s, filename)

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
        text = ht.read_close(filename)
        text = text.replace(start_tag, start_tag + '\n' + s)
        print(re.findall(r'%s'%start_tag, text))
        ht.write_close(text, filename)

    def gd_adjoint(
            filename, 
            n=5, 
            dx=1.0, 
            dz=1.0, 
            kx=3, 
            ky=3
            ):
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
    
    def make_ricker(
            nt, 
            dt, 
            f, 
            filename='OUTPUT_FILES/ricker.npy'
            ):
        t = np.linspace(0,(nt-1)*dt, nt)
        tmp1 = (1.0 - 2 * np.pi * f**2 * t**2)
        tmp2 = np.exp(-np.pi**2 * f**2 * t**2)
        np.save(filename, tmp1*tmp2)

    def src_grad(
            filenames, 
            sd, 
            dt, 
            n=5, 
            dx=1.0, 
            dz=1.0, 
            ricker_file='OUTPUT_FILES/ricker.npy',
            output_file='OUTPUT_FILES/src_grad.npy'
            ):
        s = ht.gd_adjoint(filenames, n, dx, dz)
        M = np.array([ [sd['Mxx'][0], sd['Mxz'][0]], [sd['Mxz'][0], sd['Mzz'][0]] ])
        g = np.load(ricker_file)
        scaled = g * np.matmul(M,s)
        integral = dt * np.trapz(scaled, axis=1)
        np.save(output_file, integral)
        return integral

    def run_simulator(
            mode, 
            **kw
            ):
        output_name = kw.get('output_name', 'OUTPUT_FILES.syn.adjoint')
        s = ''
        if( mode.lower()[0] == 'f' ):
            if( 'output_name' not in kw.keys() ):
                output_name = 'OUTPUT_FIILES.syn.forward'
            s = '''
                echo
                echo "running data forward simulation"
                echo
                ./change_simulation_type.pl -f

                # saving model files
                #sed -i '' "s/^SAVE_MODEL .*=.*/SAVE_MODEL = gll/" DATA/Par_file

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
        elif( mode.lower()[0] == 'a' ):
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
        print('%s\nSIMULATOR CALLED\n%s'%(80*'*',80*'*'), file=sys.stderr)
        os.system(s)

    def backtrack_and_update(
            g, 
            src_param, 
            misfit_type='l2', 
            c_armijo=0.01, 
            alpha0=2.0, 
            max_backtrack=25, 
            src_file='DATA/SOURCE', 
            data_dir='OUTPUT_FILES.dat.forward', 
            out_dir='OUTPUT_FILES.syn.backtrack', 
            final_dir='OUTPUT_FILES.syn.forward'
            ):
        xs_orig = src_param['xs'][0]
        zs_orig = src_param['zs'][0]
        phi_prime0 = -np.linalg.norm(g)**2
        alpha = alpha0
        misfitx = [float(e) for e in \
            ht.read_close('misfitx.log').split('\n') if e != ''][-1]
        misfitz = [float(e) for e in \
            ht.read_close('misfitz.log').split('\n') if e != ''][-1]
        ref_misfit = misfitx + misfitz
        misfit = np.inf
        stars = 80 * '*'
        armijo_threshold = lambda val: ref_misfit + c_armijo * alpha * phi_prime0
        curr = 0
        while( misfit > armijo_threshold(alpha) or curr > max_backtrack):
            alpha = alpha / 2.0
            print(stars)
            xs = xs_orig + alpha * g[0]
            zs = zs_orig + alpha * g[1]
            ht.update_source(xs,zs,src_file)
            ht.run_simulator('forward', output_name=out_dir)
            adj_seismogram(
                '%s/Ux_file_single_d.su'%out_dir, 
                '%s/Ux_file_single_d.su'%data_dir,
                misfit_type,
                '%s/misfitx.log'%out_dir)
            adj_seismogram(
                '%s/Uz_file_single_d.su'%out_dir, 
                '%s/Uz_file_single_d.su'%data_dir,
                misfit_type,
                '%s/misfitz.log'%out_dir)
            misfit = ht.get_last('%s/misfitx.log'%out_dir) \
                + ht.get_last('%s/misfitz.log'%out_dir)
            print(
                '(curr, g, proposed) = (%s,%s,%s)'%(
                    [xs_orig,zs_orig],
                    g,
                    [xs,zs]), 
                file=sys.stderr)
            print(
                '(alpha, misfit, threshold) = (%.8e, %.8e, %.8e)'%(
                    alpha, 
                    misfit, 
                    armijo_threshold(alpha)), 
                file=sys.stderr)
            curr += 1
        print('Successfully backtracked! alpha=%.2e'%(alpha))
        print('Moving backtrack directory "%s" to "%s"'%(out_dir, final_dir))
        os.system('rm -rf %s'%final_dir)
        os.system('mv %s %s'%(out_dir, final_dir))
        ht.append_close(
            '%.8f'%ht.get_last('%s/misfitx.log'%out_dir), 
            'misfitx.log')
        ht.append_close(
            '%.8f'%ht.get_last('%s/misfitz.log'%out_dir), 
            'misfitz.log')
        return xs,zs,alpha

    def rec_discern(
            ref_filename='DATA/Par_file_ref',
            filename='DATA/Par_file'
            ):      
        u1 = ht.par_pull(ref_filename)
        u2 = ht.par_pull(filename)
        real_receiver_no = sum(u1['nrec'])
        total_receiver_no = sum(u2['nrec'])
        return real_receiver_no, total_receiver_no
        
    def make_gif(
            x,
            y,
            name,
            fps=10,
            title_seq=lambda i : 'Frame %d'%i,
            verbose=True
            ):
        # Create a list to store frames
        frames = []

        # Loop through the data and create frames
        for i in range(len(y)):
            if( verbose ):
                print('Processing Frame %d of %d'%(i, len(y)))
            # Plot the data
            plt.plot(x, y[i])
            plt.title(title_seq(i))
            plt.savefig('%d.png'%i)
            plt.clf()

            frames.append(imageio.v2.imread('%d.png'%i))

        # Use imageio to create a GIF animation
        imageio.mimsave('%s.gif'%name.replace('.gif',''), frames, fps=fps)
        os.system('rm *.png')

    def run_convex(args):
        os.system('cp DATA/Par_file_ref DATA/Par_file')
        os.system('cp DATA/SOURCE_REF DATA/SOURCE')
        nt = par_og['NSTEP'][0]
        N = args.num_sources
        a_x = 1000
        b_x = 3000
        a_z = 500
        b_z = 2500   
        sources_x = np.linspace(a_x, b_x, N)
        sources_z = np.linspace(a_z, b_z, N)
        ref_x = sources_x[N // 2]
        ref_z = sources_z[N // 2]
        print('REF = (%f, %f)'%(ref_x,ref_z))
        lf = open(args.log, 'w')
        real_no, total_no = ht.rec_discern()
        if( args.recompute ):
            os.system('rm misfit*.log')
            
            ref_folder = 'convex_reference'
            x_suffix = 'Ux_file_single_d.%s'%args.ext
            z_suffix = 'Uz_file_single_d.%s'%args.ext
            ref_x_file = '%s/%s'%(ref_folder, x_suffix)
            ref_z_file = '%s/%s'%(ref_folder, z_suffix)
            
            prefix = 'convex'
            fldr = lambda i,j: '%s_%d_%d'%(prefix, i,j)
            get_file = lambda i,j,c: '%s/%s'%(fldr(i,j), x_suffix.replace(
                'x',c))
            
            ht.update_source(ref_x, ref_z)
            ht.run_simulator('forward', output_name=ref_folder)
            
            hf = helper()
            
            if( args.ext == 'su' ):
                data_x = hf.read_SU_file(ref_x_file)
                data_z = hf.read_SU_file(ref_z_file)
            elif( args.ext == 'bin' ):
                # data_x = hf.read_binary_file_custom_real_array(ref_x_file)
                # data_z = hf.read_binary_file_custom_real_array(ref_z_file)
                data_x = hf.read_SU_file(ref_x_file)
                data_z = hf.read_SU_file(ref_z_file)
            else:
                raise ValueError('Data ext "%s" unsupported'%args.ext)
            
            data_x = data_x[:real_no]
            data_z = data_z[:real_no]
            
            indices = range(real_no)
            
            data_x = data_x[indices]
            data_z = data_z[indices]
            
            for (i,ex) in enumerate(sources_x):
                for (j,ez) in enumerate(sources_z):
                    print('(x,z) = (%f,%f)'%(ex,ez))
                    print('(x,z) = (%f,%f)'%(ex,ez), file=lf, flush=True)
                    folder = fldr(i,j)
                    if( args.rerun ):
                        os.system('rm -rf %s'%folder)
                        os.system('mkdir -p %s'%folder)
                        ht.sco('echo "%d,%d,%.8e,%.8e" > %s/params.txt'%(
                            i,
                            j,
                            ex,
                            ez,
                            folder
                            )
                        )
                        ht.update_source(ex, ez)
                        ht.run_simulator('forward', output_name=folder)
                    syn_x = hf.read_SU_file(get_file(i,j,'x'))[indices]
                    syn_z = hf.read_SU_file(get_file(i,j,'z'))[indices]
                    if( not args.store_all ):
                        os.system(
                            'find %s ! -name "*.su" -type f -delete'%(
                                folder)
                        )
                    eval_misfit(
                        syn_x,
                        data_x,
                        nt,
                        mode=args.misfit, 
                        output='misfitx.log',
                        omit_after=real_no,
                        restrict=args.restrict)
                    eval_misfit(
                        syn_z,
                        data_z,
                        nt,
                        mode=args.misfit,
                        output='misfitz.log',
                        omit_after=real_no,
                        restrict=args.restrict)
        misfit_x = np.array(
            ht.read_close('misfitx.log').split('\n')[:-1],
            dtype=float)
        misfit_z = np.array(
            ht.read_close('misfitz.log').split('\n')[:-1],
            dtype=float)
        misfits = misfit_x + misfit_z
        misfits = misfits.reshape((N,N))
        fig, ax = plt.subplots()
        im = ax.imshow(misfits, origin='upper', extent=[a_x,b_x,a_z,b_z])
        plt.colorbar(im)
        plt.savefig('%s/%s_%d_%d_%d_%d_%d.pdf'%(
            ref_folder,
            args.misfit,
            N,
            int(a_x),
            int(b_x),
            int(a_z),
            int(b_z)          
            )
        )

    def build_execution_folder(**kw):
        target = kw['target']
        xs = kw['xs']
        zs = kw['zs']
        reference = os.getcwd()
        os.system('mkdir -p %s'%target)
        os.chdir(target)
        os.system('ln -s %s/*.py 2> /dev/null'%reference)
        os.system('ln -s %s/run_this_example.sh'%reference)
        os.system('ln -s %s/xmeshfem2D'%reference)
        os.system('ln -s %s/xspecfem2D'%reference)
        os.system('cp -r %s/DATA .'%reference)
        ht.update_source(xs,zs)
        os.chdir(reference)

    def build_tree(**kw):
        os.system('cp DATA/Par_file_ref DATA/Par_file')
        os.system('cp DATA/SOURCE_REF DATA/SOURCE')
        N = kw['N']
        ax = kw['ax']
        bx = kw['bx']
        az = kw['az']
        bz = kw['bz']  
        sources_x = np.linspace(ax, bx, N)
        sources_z = np.linspace(az, bz, N)
        ref_x = sources_x[N // 2]
        ref_z = sources_z[N // 2]
        print('REF = (%f, %f)'%(ref_x,ref_z))
        prefix = 'convex'
        ref_folder = '%s_reference'%prefix
        fldr = lambda i,j: '%s_%d_%d'%(prefix, i,j)
        
        ht.build_execution_folder(target=ref_folder, xs=ref_x, zs=ref_z)
        
        hf = helper()
        
        folders = [ref_folder]
        for (i,ex) in enumerate(sources_x):
            for (j,ez) in enumerate(sources_z):
                print('(x,z) = (%f,%f)'%(ex,ez))
                folder = fldr(i,j)
                os.system('rm -rf %s'%folder)
                os.system('mkdir -p %s'%folder)
                ht.build_execution_folder(target=folder, xs=ex, zs=ez)
                folders.append(folder)
        return folders
    
    def execute_node(folder, c='&'):
        # print('LAUNCH %s'%folder, file=sys.stderr)
        # cmd = 'cd %s; ./run_this_example.sh > run.out 2> run.err'%folder
        # ht.sco(cmd)
        reference = os.getcwd()
        os.chdir(folder)
        os.system('./run_this_example.sh > run.out 2> run.err %s'%c)
        os.chdir(reference)

    def purge(done=set([]), save_param=False):
        if( type(done) == list ): done = set(done)
        finished_dirs = set(['/'.join(e.split('/')[:-1]) for e in \
            ht.sco('find $(pwd) -name "*.su"', True)]
        )
        new_dirs = finished_dirs.difference(done)
        mother_dir = ht.sco('echo "$SPEC_APP"', True)[0]
        for case in new_dirs:
            go_up = case.replace('/OUTPUT_FILES', '')
            if( 
                go_up == mother_dir or \
                not os.path.exists('%s/OUTPUT_FILES'%go_up) 
            ):
                continue
            if( save_param ):
                ht.sco('mv -n %s/DATA/Par_file %s/SOURCE.su'%(go_up, go_up))
                ht.sco('mv -n %s/DATA/Par_file %s/SOURCE.su'%(go_up, go_up))
            ht.sco('mv -n %s/*.su %s'%(case, go_up))
            ht.sco('rm -rf %s'%case)
            ht.sco('rm -rf %s/DATA'%go_up)
            ht.sco('find %s ! -type d ! -name "*.su" -exec rm {} +'%go_up)
        return finished_dirs


if( __name__ == "__main__" ):
    try:
        parser = argparse.ArgumentParser(
            description="Driver for source inversion")
        
        parser.add_argument(
            "mode", 
            type=int, 
            help="mode of execution (int: 1 <= mode <= 8)")
        parser.add_argument(
            "--misfit", 
            default="l2", 
            type=str,
            help="Misfit functional, either l2 or w2")
        parser.add_argument(
            "--plot", 
            action='store_true', 
            help="Perform seismogram plots")
        parser.add_argument(
            "--num_sources", 
            default=10, 
            type=int,
            help="Convexity plot granularity")
        parser.add_argument(
            '--rerun', 
            action='store_true',
            help="Rerun forward solving")
        parser.add_argument(
            '--recompute',
            action='store_true',
            help='Recompute misfit')
        parser.add_argument(
            '--restrict',
            default=None,
            type=float,
            help='Wasserstein restriction')
        parser.add_argument(
            '--ext',
            default='su',
            help='Data file extension'
        )
        parser.add_argument(
            '--log',
            default='logger.log',
            help='logger file'
        )
        parser.add_argument(
            '--store_all',
            action='store_true',
            help='Store all files'
        )
        parser.add_argument(
            '--max_proc',
            default=16,
            type=int,
            help='Max number of spawned processes at one time'
        )
        parser.add_argument(
            '--purge_interval',
            default=100,
            type=int,
            help='Every purge_interval steps, extra files eliminated'
        )
        args = parser.parse_args()
        
        mode = args.mode
        
        print('Run case: %s'%str(args))
        src_og = ht.src_pull()
        par_og = ht.par_pull()

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
            v = ht.src_pull()
            src = [v['xs'][0], v['zs'][0]]
            ht.add_artificial_receivers(src, par_og['nreceiversets'][0])
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
            ht.add_artificial_receivers([xs,zs], par_og['nreceiversets'][0], filename='DATA/Par_file')
        elif( mode == 7 ):
            print('\n\n\n%s\nBEGINNING BACKTRACK\n%s\n\n\n'%(80*'*',80*'*'), file=sys.stderr)
            os.system('sleep 3')
            print('...')
            nt = par_og['NSTEP'][0]
            dt = par_og['DT'][0]
            freq = src_og['f0'][0]
            ht.make_ricker(nt,dt,freq)

            # filenames = ['SEM/Ux_file_single.su.adj', 
            #     'SEM/Uz_file_single.su.adj']
            filenames = ['OUTPUT_FILES.syn.adjoint/Ux_file_single_d.su',
                'OUTPUT_FILES.syn.adjoint/Uz_file_single_d.su']
            src_curr = ht.src_pull()
            recs = 5
            g = ht.src_grad(filenames, src_curr, dt, n=recs)
            src_curr['nreceiversets'] = recs + par_og['nreceiversets'][0]
            ht.backtrack_and_update(g, src_curr, 
                misfit_type=sys.argv[2].lower(), 
                c_armijo=0.0001, 
                alpha0=2.0)
        elif( mode == 8 ):
            os.system('cp DATA/Par_file_ref DATA/Par_file')
            os.system('cp DATA/SOURCE_REF DATA/SOURCE')
            nt = par_og['NSTEP'][0]
            N = args.num_sources
            a_x = 1000
            b_x = 3000
            a_z = 500
            b_z = 2500   
            sources_x = np.linspace(a_x, b_x, N)
            sources_z = np.linspace(a_z, b_z, N)
            ref_x = sources_x[N // 2]
            ref_z = sources_z[N // 2]
            print('REF = (%f, %f)'%(ref_x,ref_z))
            lf = open(args.log, 'w')
            real_no, total_no = ht.rec_discern()
            if( args.recompute ):
                os.system('rm misfit*.log')
                
                ref_folder = 'convex_reference'
                x_suffix = 'Ux_file_single_d.%s'%args.ext
                z_suffix = 'Uz_file_single_d.%s'%args.ext
                ref_x_file = '%s/%s'%(ref_folder, x_suffix)
                ref_z_file = '%s/%s'%(ref_folder, z_suffix)
                
                prefix = 'convex'
                fldr = lambda i,j: '%s_%d_%d'%(prefix, i,j)
                get_file = lambda i,j,c: '%s/%s'%(fldr(i,j), x_suffix.replace(
                    'x',c))
                
                ht.update_source(ref_x, ref_z)
                ht.run_simulator('forward', output_name=ref_folder)
                
                hf = helper()
                
                if( args.ext == 'su' ):
                    data_x = hf.read_SU_file(ref_x_file)
                    data_z = hf.read_SU_file(ref_z_file)
                elif( args.ext == 'bin' ):
                    # data_x = hf.read_binary_file_custom_real_array(ref_x_file)
                    # data_z = hf.read_binary_file_custom_real_array(ref_z_file)
                    data_x = hf.read_SU_file(ref_x_file)
                    data_z = hf.read_SU_file(ref_z_file)
                else:
                    raise ValueError('Data ext "%s" unsupported'%args.ext)
                
                data_x = data_x[:real_no]
                data_z = data_z[:real_no]
                
                indices = range(real_no)
                
                data_x = data_x[indices]
                data_z = data_z[indices]
                
                total_time = 0.0
                for (i,ex) in enumerate(sources_x):
                    for (j,ez) in enumerate(sources_z):
                        print('(x,z) = (%f,%f)'%(ex,ez))
                        print('(x,z) = (%f,%f)'%(ex,ez), file=lf, flush=True)
                        folder = fldr(i,j)
                        if( args.rerun ):
                            os.system('rm -rf %s'%folder)
                            os.system('mkdir -p %s'%folder)
                            ht.sco('echo "%d,%d,%.8e,%.8e" > %s/params.txt'%(
                                i,
                                j,
                                ex,
                                ez,
                                folder
                                )
                            )
                            ht.update_source(ex, ez)
                            tmp = time.time()
                            ht.run_simulator('forward', output_name=folder)
                            total_time += time.time() - tmp
                        syn_x = hf.read_SU_file(get_file(i,j,'x'))[indices]
                        syn_z = hf.read_SU_file(get_file(i,j,'z'))[indices]
                        if( not args.store_all ):
                            os.system(
                                'find %s ! -name "*.su" -type f -delete'%(
                                    folder)
                            )
                        eval_misfit(
                            syn_x,
                            data_x,
                            nt,
                            mode=args.misfit, 
                            output='misfitx.log',
                            omit_after=real_no,
                            restrict=args.restrict)
                        eval_misfit(
                            syn_z,
                            data_z,
                            nt,
                            mode=args.misfit,
                            output='misfitz.log',
                            omit_after=real_no,
                            restrict=args.restrict)
            misfit_x = np.array(
                ht.read_close('misfitx.log').split('\n')[:-1],
                dtype=float)
            misfit_z = np.array(
                ht.read_close('misfitz.log').split('\n')[:-1],
                dtype=float)
            misfits = misfit_x + misfit_z
            misfits = misfits.reshape((N,N))
            fig, ax = plt.subplots()
            im = ax.imshow(misfits, origin='upper', extent=[a_x,b_x,a_z,b_z])
            plt.colorbar(im)
            plt.savefig('%s/%s_%d_%d_%d_%d_%d.pdf'%(
                ref_folder,
                args.misfit,
                N,
                int(a_x),
                int(b_x),
                int(a_z),
                int(b_z)          
                )
            )
            print('TOTAL SERIAL FORWARD: %.2f'%total_time)
        elif( mode == 9 ):
            hf = helper()
            x_data = []
            z_data = []
            if( args.rerun ):
                x_files = ht.sco('find convex* -name "Ux_file*.su"', True)
                z_files = ht.sco('find convex* -name "Uz_file*.su"', True)
                for xf in x_files:
                    x_data.append(hf.read_SU_file(xf))
                    prefix = '/'.join(xf.split('/')[:-1])
                    suffix = 'ux.npy'
                    np.save('%s/%s'%(prefix,suffix), x_data[-1])
                for zf in z_files:
                    z_data.append(hf.read_SU_file(zf))
                    prefix = '/'.join(zf.split('/')[:-1])
                    suffix = 'uz.npy'
                    np.save('%s/%s'%(prefix,suffix), z_data[-1])
            else:
                x_files = ht.sco('find convex* -name "ux.npy"', True)
                z_files = ht.sco('find convex* -name "uz.npy"', True)
                [x_data.append(np.load(f)) for f in x_files]
                [z_data.append(np.load(f)) for f in z_files]
            x_data = np.transpose(x_data,axes=(0,1,2))
            z_data = np.transpose(z_data,axes=(0,1,2))
            dt = par_og['DT'][0]
            t = [i * dt for i in range(x_data.shape[-1])]
            src_order = ht.sco(
                'ls -tr | grep "convex_[0-9]_*" | sed "s/convex_//"', 
                True)
            for i in range(x_data.shape[0]):
                curr_x = x_data[i]
                curr_z = z_data[i]
                print('Processing gif %d'%i)
                ht.make_gif(
                    t, 
                    curr_x,
                    'x_traces_%s.gif'%src_order[i], 
                    title_seq=lambda j : 'Receiver %d'%j,
                    verbose=False)
                ht.make_gif(
                    t, 
                    curr_z,
                    'z_traces_%s.gif'%src_order[i],
                    title_seq=lambda j : 'Receiver %d'%j, 
                    verbose=False)
        elif( mode == 10 ):
            t_orig = time.time()
            folders = ht.build_tree(
                N=args.num_sources,
                ax=1000,
                bx=3000,
                az=500,
                bz=2500
            )
            print('Folder build time: %.2f'%(time.time() - t_orig))
            max_proc = args.max_proc
            done = set([])
            save_param = False
            purge_interval = args.purge_interval
            purge_time = 0.0
            t = time.time()
            for (i,folder) in enumerate(folders):
                if( np.mod(i+1, max_proc) == 0 ):
                    ht.execute_node(folder, '')
                    print('(batch,global)=(%d,%d)'%(i//max_proc,i))
                else:
                    ht.execute_node(folder, '&')
                if( np.mod(i+1, purge_interval) == 0 ):
                    purge_tmp = time.time()
                    done = ht.purge(done, save_param)
                    purge_time += time.time() - purge_tmp
            print('Launch time: %.2f'%(time.time() - t - purge_time))
            print('Purge time: %.2f'%purge_time)
            t = time.time()
            left = lambda : len(
                [e for e in ht.sco('ps', True) if 'xspecfem2D' in e]
            )
            orig = args.num_sources**2 + 1
            curr = orig
            while( curr == 0 ): 
                print('Waiting for process delay...')
                curr = left()
            while( curr > 0 ): 
                print('TOTAL TIME: %.2f...%d/%d remaining'%(
                    time.time() - t,
                    curr,
                    orig)
                )
                try:
                    done = ht.purge(done, save_param)
                except:
                    print('Something wrong in purge')
                time.sleep(5)
                curr = left()
            try:
                ht.purge(done, save_param)
            except:
                print('Double check that nothing went wrong lol')
            print('COMPLETE RUN TIME = %.2f'%(time.time() - t_orig))
    except Exception as e:
        traceback.print_exc()
        exit(-1)





