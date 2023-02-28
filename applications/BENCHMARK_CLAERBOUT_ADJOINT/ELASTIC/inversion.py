import numpy as np
from subprocess import check_output as co
import os
import sys
import time

def postprocess(files, output_name='MOVIES'):
    spec_app = '/Users/tyler/spec/specfem2d/SpecFEMInversion/' \
        + 'applications/BENCHMARK_CLAERBOUT_ADJOINT/ELASTIC'
    store = 'FIGURES'
    print(spec_app)
    s = '%s/%s'%(spec_app, store)
    os.system('mkdir -p %s'%s)
    try:
        next = 1 + int(co('ls -t %s | head -n 1'%s, shell=True).decode('utf-8'))
    except:
        next = 1
    latest = '%s/%d'%(s,next)
    os.system('cp -rp %s/FINAL_RESULTS %s'%(spec_app, latest))
    os.system('mkdir -p %s/%s'%(latest, output_name))
    for base, ext in files:
        cmd = 'convert -delay 20 -loop 0 $(find %s -name "*%s.%s") %s'%(
            latest, base, ext, '%s/%s/%s.gif'%(latest, output_name, base))
        print(cmd)
        os.system(cmd)

if( __name__ == "__main__" ):
    very_start = time.time()
    print('CLEARING PAST RESULTS')
    os.system('./clear_dir.sh')
    print('SETTING UP CURRENT RUN')
    os.system('./setup.sh > DATA/setup.log')
    os.system('echo "%d seconds for setup" > DATA/setup.log'%(time.time() - very_start))

    misfits = [np.inf]
    max_iter = 5
    tol = 0.0
    iter = 1
    mode = 'l2' if len(sys.argv) < 2 else sys.argv[1]
    save_every = 1
    verbose = True

    dirs = ['OUTPUT_FILES.syn.forward', 
        'OUTPUT_FILES.syn.adjoint', 
        'KERNELS',
        'DATA']

    os.system('rm -rf FINAL_RESULTS')
    os.system('mkdir -p FINAL_RESULTS')

    print('BEGINNING INVERSION')

    running_total = 0.0
    while( misfits[-1] > tol and iter <= max_iter ):
        t = time.time()
        print('ITERATION %d'%iter)
        os.system('mkdir FINAL_RESULTS/%d'%(iter))
        os.system('sleep 1')
        if( verbose ):
            os.system('./run_adjoint.sh %d %s'%(iter, mode))
        else:
            os.system('./run_adjoint.sh %d %s > OUTPUT_FILES/%d.log'%(iter, mode, iter))
        misfitx = float(co('cat misfitx.log | tail -n 1', shell=True).decode('utf-8'))
        misfitz = float(co('cat misfitz.log | tail -n 1', shell=True).decode('utf-8'))
        misfits.append(misfitx + misfitz)
        if( np.mod(iter, save_every) == 0 ):
            [os.system("cp -r %s FINAL_RESULTS/%d"%(d,iter)) for d in dirs]
        curr_time = time.time() - t
        running_total += curr_time
        print('(i,misfit,tc,tra,ETR) = (%d,%.2e,%f,%f,%f)'%(iter, 
            misfits[-1],
            curr_time,
            running_total / iter,
            (running_total / iter) * (max_iter - iter)))
        iter += 1

    postprocess([('vp_new', 'jpg'), \
        ('vs_new', 'jpg'), \
        ('rho_new', 'jpg')])

    print('INVERSION SUCCESS!!! INFO BELOW\n%s'%(80 * '*'))
    print('ITERATIONS: %d'%(iter - 1))
    print('FINAL MISFIT: %.2e'%(misfits[-1]))
    print('AVG TIME PER LOOP: %.2e seconds'%(running_total / (iter-1)))
    print('TOTAL RUN TIME OF INNER LOOP: %.2e seconds'%(running_total))
    print('TOTAL RUN TIME: %.2e seconds'%(time.time() - very_start))

postprocess([('vp_new', 'jpg'), \
    ('vs_new', 'jpg'), \
    ('rho_new', 'jpg')])