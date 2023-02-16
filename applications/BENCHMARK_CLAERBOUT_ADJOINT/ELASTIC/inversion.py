import numpy as np
from subprocess import check_output as co
import os
import sys
import time

t = time.time()
os.system('./setup.sh > DATA/setup.log')
os.system('echo "%d seconds for setup" > DATA/setup.log'%(t - time.time()))

misfits = [np.inf]
max_iter = 2
tol = 1e-15
iter = 1
while( misfits[-1] > tol and iter <= max_iter ):
    print('ITERATION %d'%iter)
    os.system('sleep 1')
    os.system('./run_adjoint.sh')
    misfitx = float(co('cat misfitx.log | tail -n 1', shell=True).decode('utf-8'))
    misfitz = float(co('cat misfitz.log | tail -n 1', shell=True).decode('utf-8'))
    misfits.append(misfitx + misfitz)
    iter += 1
