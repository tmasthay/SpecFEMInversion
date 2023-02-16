import numpy as np
from subprocess import check_output as co
import os
import sys
import time

very_start = time.time()
print('CLEARING PAST RESULTS')
os.system('./clear_dir.sh')
print('SETTING UP CURRENT RUN')
os.system('./setup.sh > DATA/setup.log')
os.system('echo "%d seconds for setup" > DATA/setup.log'%(time.time() - very_start))

misfits = [np.inf]
max_iter = 2
tol = 0.0
iter = 1
mode = 'l2' if len(sys.argv) < 2 else sys.argv[1]
save_every = 1

dirs = ['OUTPUT_FILES.syn.forward', 'OUTPUT_FILES.syn.adjoint']

os.system('rm -rf FINAL_RESULTS')
os.system('mkdir -p FINAL_RESULTS')

print('BEGINNING INVERSION')

running_total = 0.0
while( misfits[-1] > tol and iter <= max_iter ):
    t = time.time()
    print('ITERATION %d'%iter)
    os.system('mkdir FINAL_RESULTS/%d'%(iter))
    os.system('sleep 1')
    os.system('./run_adjoint.sh %d %s'%(iter, mode))
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
        (running_total / iter) * (max_iter - iter - 1)))
    iter += 1

print('INVERSION SUCCESS!!! INFO BELOW\n%s'%(80 * '*'))
print('ITERATIONS: %d'%(iter - 1))
print('FINAL MISFIT: %.2e'%(misfits[-1]))
print('AVG TIME PER LOOP: %.2e seconds'%(running_total / (iter-1)))
print('TOTAL RUN TIME OF INNER LOOP: %.2e seconds'%(running_total))
print('TOTAL RUN TIME: %.2e seconds'%(time.time() - very_start))