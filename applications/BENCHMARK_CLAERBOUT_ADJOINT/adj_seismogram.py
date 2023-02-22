#!/usr/bin/env python
#
# creates adjoint sources f^adj = (syn - data)
#
#########################################
from __future__ import print_function

import sys
import numpy as np
import array
import os

from helper_functions import helper

from wasserstein import *
from subprocess import check_output as co
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from helper_tyler import *

#########################################

## globals
# takes 2nd-derivative of pressure for adjoint source
use_derivative_of_pressure = True

#########################################


def adj_seismogram_get_files(NSTEP,DT,NPROC,SIM_TYPE):
    # user output
    print("creating adjoint seismograms:")
    print("  NSTEP = ",NSTEP)
    print("  DT    = ",DT)
    print("  NPROC = ",NPROC)
    print("  type  = ",SIM_TYPE,"(acoustic == 1 / elastic P-SV == 2 / elastic SH == 3)")
    print("")

    # selects components
    if SIM_TYPE == 1:
        # acoustic
        comp_total = [0] # single trace
    elif SIM_TYPE == 2:
        # elastic
        # P-SV
        comp_total = [1,3]  # Ux/Uz
    elif SIM_TYPE == 3:
        # elastic
        # SH
        comp_total = [2]  # Uy
    else:
        print("Invalid SIM_TYPE {} for selecting component".format(SIM_TYPE))
        sys.exit(1)

    # user output
    print("seismograms: ")
    for icomp  in comp_total:
        name = "file_single"
        if icomp == 0:
            # acoustic
            # for example: Up_file_single_p.su
            # pressure-component
            filename_in  = "Up_" + name + "_p.su"  # from acceleration
            filename_out = "Up_" + name + ".su.adj"  # adjoint trace assumes Up_file_single.su name for acoustic simulations (pressure/potential type)
        elif icomp == 1:
            # elastic
            # for example P-SV case: Ux_file_single_d.su and Uz_file_single_d_su
            # x-component (P-SV case)
            filename_in  = "Ux_" + name + "_d.su" # from displacement
            filename_out = "Ux_" + name + ".su.adj" # adjoint traces assumes Ux_file_single.su.adj name
        elif icomp == 2:
            # y-component (SH-case)
            filename_in  = "Uy_" + name + "_d.su"
            filename_out = "Uy_" + name + ".su.adj" # adjoint traces assumes Uy_file_single.su.adj
        elif icomp == 3:
            # z-component (P-SV case)
            filename_in  = "Uz_" + name + "_d.su"
            filename_out = "Ux_" + name + ".su.adj" # adjoint traces assumes Uz_file_single.su.adj
        else:
          print("Invalid component ",icomp)
          sys.exit(1)

        filename_syn = "SEM/syn/" + filename_in
        filename_dat = "SEM/dat/" + filename_in
        filename_adj = "SEM/" + filename_out


#
#------------------------------------------------------------------------------------------
#

def adj_seismogram(filename_syn,filename_dat, mode='w2', output='misfitx.log', **kw):
    """
    creates adjoint seismograms
    """
    global use_derivative_of_pressure

    print("")
    print("creating adjoint seismograms:")
    print("  input syn : ",filename_syn)
    print("  input data: ",filename_dat)
    print("")

    # helper functions
    hf = helper()

    # basename
    name = os.path.basename(filename_syn)

    # reads in seismograms
    syn = hf.read_SU_file(filename_syn)
    dat = hf.read_SU_file(filename_dat)

    omit = kw.get('artificial', 0)
    
    # sampling rate given in microseconds
    DT = hf.sampling_DT * 1.e-6
    print("  trace time steps: DT = ",DT,"(s)")

    output_file = open(output, 'a')

    # adjoint source f^adj = (s - d)
    if( mode.lower() == 'l2' ):
        adj = syn - dat
        if( omit > 0 ):
            adj[(len(adj) - omit):] = 0.0
        # misfit values
        print("misfit:")
        diff_max = np.abs(adj).max()
        print("  maximum adjoint value (syn - dat) = ",diff_max)

        # total misfit
        total_misfit = 0.0
        for irec in range(len(adj)):
            # single receiver trace
            adj_trace = adj[irec]
            # inner product
            total_misfit += np.sum(adj_trace * adj_trace) * DT

        print("")
        output_file.write("{:e}\n".format(total_misfit))
        print("{:e}".format(total_misfit))
        print("")

        # number of receivers/traces (SU files have only single components)
        num_receivers = len(adj)

        print("adjoint source:")
        print("  number of traces = ",num_receivers)

        # checks
        if num_receivers == 0:
            print("Did find no receivers or traces, please check...")
            sys.exit(1)

        # for acoustic FWI, L2 adjoint source is the second derivative of pressure difference
        # (e.g., see Peter et al. 2011, GJI, eq. (A8))
        if name == "Up_file_single_p.su":
            # pressure output
            # note: pressure in fluid is defined as p = - \partial_t^2 phi
            #       thus, if potential phi is chosen as output, there is a minus sign and time derivative difference.
            #
            #       assuming a pressure output for syn and dat, the adjoint source expression is given by (A8) in Peter et al. (2011)
            #       note the negative sign in the definition.
            #       the adjoint source for pressure is: f^adj = - \partial_t^2 p_syn - \partial_t^2 p_obs
            #                                                 = - \partial_t^2 ( p_syn - p_obs )
            #
            if use_derivative_of_pressure:
                print("  creating adjoint sources for pressure (taking second-derivative of pressure differences)...")
                # takes second-derivative
                adj_new = adj.copy()
                fac = 1.0 / DT**2
                for irec in range(num_receivers):
                    # single receiver trace
                    adj_trace = adj[irec]
                    for i in range(1,len(adj_trace)-1):
                        # do a simple central finite difference
                        val = (adj_trace[i+1] - 2.0 * adj_trace[i] + adj_trace[i-1]) * fac
                        # adding negative sign
                        adj_new[irec][i] = - val

                # saves as adjoint source
                adj = adj_new.copy()
    elif( mode.lower() == 'w2' ):
        restrict = 0.1
        nt = int(co('cat DATA/Par_file | grep "NSTEP"', shell=True) \
            .decode('utf-8').split('=')[-1])
        i1,i2 = cut(nt, restrict)
        adj = np.zeros((syn.shape[0], i2-i1))
        dists = np.zeros(syn.shape[0])
        t = np.array([DT*n for n in range(i1,i2)])
        for i in range(syn.shape[0]):
            s = split_normalize(syn[i])
            #recall in future that CDF of d doesn't need to be computed every time
            d = split_normalize(dat[i])
            dists[i], adj[i], Q, D, U = wass_adjoint_and_eval(d=d,u=s,dt=DT,ot=0.0,nt=nt,restrict=restrict,multi=True)

            if( omit > 0 and i >= len(adj) - omit ):
                adj[i] = 0.0
                dists[i] = 0.0

                perform_plots = False
                if( perform_plots ):
                    plt.rcParams['text.usetex'] = True
                    plt.figure(1)
                    plt.subplots_adjust(wspace=0.4, hspace=0.4)
                    plt.subplot(2,2,1)
                    plt.plot(t, dat[i][i1:i2], linewidth=1.0, label='Observed data')
                    plt.plot(t, syn[i][i1:i2], linestyle='dashed', linewidth=0.5, label='Synthetic')
                    plt.title(r'Raw Trace Data, trace %d'%i)
                    # formatter = ScalarFormatter(useMathText=True)
                    # formatter.set_scientific(True)
                    # plt.gca().yaxis.set_major_formatter(formatter)
                    ht.sn_plot()
                    plt.legend(fontsize=8)

                    plt.subplot(2,2,2)
                    plt.scatter(t, np.log(dat[i][i1:i2]) / np.log(10.0), marker='.', s=0.2, label='Observed data')
                    plt.scatter(t, np.log(syn[i][i1:i2]) / np.log(10.0), marker='*', s=0.2, label='Synthetic')
                    plt.title(r'Log-scale Trace Data, trace %d'%i)
                    plt.legend(fontsize=8)

                    plt.subplot(2,2,3)
                    plt.plot(t,adj[i],linestyle='-',label='W2 adjoint')
                    ht.sn_plot()
                    plt.title(r'$W_2$ Adjoint for trace %d'%i)

                    plt.subplot(2,2,4)
                    plt.plot(t,syn[i][i1:i2] - dat[i][i1:i2], linestyle='-', linewidth=0.5, label='L2 adjoint')
                    ht.sn_plot()
                    plt.title('$L_2$ Adjoint for trace %d'%i)
                    plt.savefig('OUTPUT_FILES/adjoint_%d.pdf'%i)
                    plt.clf()
        output_file.write('{:e}\n'.format(sum(dists)))
        print('Total W2 misfit: %.2e'%(sum(dists)))
    else:
        raise ValueError('Mode "%s" not supported'%mode)
    output_file.close()
        
    # statistics
    amp_max = np.abs(adj).max()
    print("  maximum amplitude |f^adj| = ",amp_max)

    print("")

    # SEM output directory
    os.system("mkdir -p SEM/")

    # adjoint trace name
    if name == "Up_file_single_p.su":
        # acoustic pressure traces
        # adjoint traces assumes Up_file_single.su.adj name
        adjname = "Up_file_single.su"
    elif name == "Up_file_single_x.su":
        # acoustic potential traces
        # adjoint traces assumes Up_file_single.su.adj name
        adjname = "Up_file_single.su"
    elif name == "Ux_file_single_d.su":
        # adjoint traces assumes Ux_file_single.su.adj name
        adjname = "Ux_file_single.su"
    elif name == "Uy_file_single_d.su":
        # adjoint traces assumes Uy_file_single.su.adj name
        adjname = "Uy_file_single.su"
    elif name == "Uz_file_single_d.su":
        # adjoint traces assumes Uz_file_single.su.adj name
        adjname = "Uz_file_single.su"
    else:
        print("Error: trace name {} not recognized".format(name))
        sys.exit(1)

    filename_adj = "SEM/" + adjname + ".adj"

    hf.write_SU_file(adj,filename_adj)

    # user output
    print("  receivers: ",len(adj))
    print("  adjoint sources written to: ",filename_adj)
    print("")

#
#------------------------------------------------------------------------------------------
#

def usage():
    #print("Usage: ./adj_seismogram.py NSTEP DT NPROC type[acoustic/elastic]")
    print("Usage: ./adj_seismogram.py synthetics-SU-file data-SU-file")
    print("")
    sys.exit(1)


#
#------------------------------------------------------------------------------------------
#

if __name__ == '__main__':
    # gets arguments
    if len(sys.argv) < 3:
        usage()

    #NSTEP = int(sys.argv[1])
    #DT = float(sys.argv[2])
    #NPROC = int(sys.argv[3])
    #str_type = sys.argv[4]

    # checks
    #if NSTEP <= 0:
    #    print("Invalid NSTEP, must be > 0")
    #    sys.exit(1)
    #if NPROC <= 0:
    #    print("Invalid NPROC, must be > 0")
    #    sys.exit(1)

    #if str_type == "acoustic":
    #    SIM_TYPE = 1
    #elif str_type == "elastic":
    #    SIM_TYPE = 2
    #else:
    #    print("Error: invalid type ",str_type," must be: acoustic or elastic")
    #    sys.exit(1)

    #adj_seismogram_get_files(NSTEP,DT,NPROC,SIM_TYPE)

    filename_syn = sys.argv[1]
    filename_dat = sys.argv[2]
    mode = sys.argv[3]
    output_file = sys.argv[4]

    N = 9
    adj_seismogram(filename_syn,filename_dat,mode,output_file, artificial=N)
    #adj_seismogram(filename_syn,filename_dat,mode)

    print("")
    print("all done")
    print("")
