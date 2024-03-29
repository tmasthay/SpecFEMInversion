#!/usr/bin/env python
#
# perturbs model with a Gaussian perturbation
#
#########################################
from __future__ import print_function

import sys
import numpy as np
import array

from helper_functions import helper

##########################################################

## default setup
NGLLX = 5
NGLLZ = 5

IN_DATA_FILES = './DATA/'

# Gaussian center position
CENTER_X = 500.0
CENTER_Z = 500.0

# default pertubation wavelength
# using frequency 10 Hz (large enough to be resolved by grid and source wavelength to avoid numerical artifacts),
# vp from model in Par_file is set to 2700 m/s
wavelength = 1.0/10.0 * 2700.0

# plotting model figures
do_plot_figures = False

##########################################################


def model_add_Gaussian_perturbation(percent,NPROC,DO_PERTURB_RHO,DO_PERTURB_VP,DO_PERTURB_VS):
    """
    adds Gaussian perturbation to model
    """
    global wavelength
    global do_plot_figures

    # helper functions
    hf = helper()

    # loop over processes
    for myrank in range(0, NPROC):
        print("")
        print("reading in mesh files for rank {}...".format(myrank))
        print("")

        # processors name
        prname = IN_DATA_FILES + "proc{:06d}_".format(myrank)

        # note: we could deal with adding the Gaussian perturbation without having nspec and ibool arrays,
        #       and use just 1D-arrays read in from data files (since all positional xstore/zstore and
        #       model vpstore/.. arrays have the same dimensions (NGLLX,NGLLZ,nspec) -> 1D (NGLLX*NGLLZ*nspec) arrays)
        #       thus, we could just compute position and add Gaussian perturbations for each 1D-array entries.
        #
        #       nevertheless, having nspec & ibool might become useful for future handlings, so we'll use it
        #       here as an example use case.

        # mesh index
        filename = prname + "NSPEC_ibool.bin"

        nspec,ibool = hf.read_binary_NSPEC_ibool_file(filename=filename)

        ibool = np.reshape(ibool, (NGLLX, NGLLZ, nspec), order='F') # fortran-like index ordering
        #debug
        #print("debug: ibool", ibool[0:4,0,0])
        #for ispec  in range(0, 3):
        #    for j in range(0, NGLLZ):
        #        for i in range(0, NGLLX):
        #            # GLL point location (given in m: dimension 2640 m x 2640 x x 1440 m)
        #            iglob = ibool[i,j,ispec]
        #            print("debug: ibool i,j,ispec - iglob ",i,j,ispec," - ",iglob)

        # coordinates
        # global point arrays
        #  x_save(NGLLX,NGLLZ,nspec)
        filename = prname + "x.bin"
        xstore = hf.read_binary_SEM_file(filename)
        xstore = np.reshape(xstore, (NGLLX, NGLLZ, nspec), order='F') # fortran-like index ordering
        filename = prname + "z.bin"
        zstore = hf.read_binary_SEM_file(filename)
        zstore = np.reshape(zstore, (NGLLX, NGLLZ, nspec), order='F') # fortran-like index ordering

        # user output
        if myrank == 0:
            print("")
            print("mesh:")
            print("  nspec = ",nspec)
            print("  model dimension: x min/max = ",xstore.min(),xstore.max())
            print("                   z min/max = ",zstore.min(),zstore.max())
            print("")
            print("")

        # user output
        if myrank == 0:
            print("model: rho,vp,vs")
            print("")

        # model rho,vp,vs
        filename = prname + "rho.bin"
        rhostore = hf.read_binary_SEM_file(filename)
        rhostore = np.reshape(rhostore, (NGLLX, NGLLZ, nspec), order='F') # fortran-like index ordering
        filename = prname + "vp.bin"
        vpstore = hf.read_binary_SEM_file(filename)
        vpstore = np.reshape(vpstore, (NGLLX, NGLLZ, nspec), order='F') # fortran-like index ordering
        filename = prname + "vs.bin"
        vsstore = hf.read_binary_SEM_file(filename)
        vsstore = np.reshape(vsstore, (NGLLX, NGLLZ, nspec), order='F') # fortran-like index ordering

        # plot images
        if do_plot_figures:
            hf.plot_model_image(xstore,zstore,rhostore,prname + "rho")
            hf.plot_model_image(xstore,zstore,vpstore,prname + "vp")
            hf.plot_model_image(xstore,zstore,vsstore,prname + "vs")

        # model perturbation
        # initializes perturbations
        pert_param = np.zeros([NGLLX,NGLLZ,nspec])

        # Gaussian
        # standard variance
        sigma_h = 0.25 * wavelength / np.sqrt(8.0)      # such that scalelength becomes 1/4 of dominant wavelength
        sigma_v = 0.25 * wavelength / np.sqrt(8.0)
        # factor two for Gaussian distribution with standard variance sigma
        sigma_h2 = 2.0 * sigma_h * sigma_h
        sigma_v2 = 2.0 * sigma_v * sigma_v

        # scalelength: approximately S ~ sigma * sqrt(8.0) for a Gaussian smoothing
        if myrank == 0:
            print("")
            print("Gaussian perturbation:")
            print("  input wavelength                       (m): ",wavelength)
            print("  using scalelengths horizontal,vertical (m): ",sigma_h*np.sqrt(8.0),sigma_v*np.sqrt(8.0))
            print("")
            print("  pertubation center location         : x/z = ",CENTER_X,"/",CENTER_Z)
            print("  perturbation strength               : ",percent)
            if DO_PERTURB_RHO : print("  perturbing                          : rho values")
            if DO_PERTURB_VP  : print("  perturbing                          : Vp values")
            if DO_PERTURB_VS  : print("  perturbing                          : Vs values")
            print("")
            print("rank ",myrank,":")
            print("  initial model velocity  : rho min/max = ",rhostore.min(),rhostore.max())
            print("                            vp  min/max = ",vpstore.min(),vpstore.max())
            print("                            vs  min/max = ",vsstore.min(),vsstore.max())
            print("")

        # theoretic normal value
        # (see integral over -inf to +inf of exp[- x*x/(2*sigma) ] = sigma * sqrt(2*pi) )
        # note: smoothing is using a Gaussian (ellipsoid for sigma_h  !=  sigma_v),
        #norm_h = 2.0*PI*sigma_h**2
        #norm_v = sqrt(2.0*PI) * sigma_v
        #norm   = norm_h * norm_v

        # sets Gaussian perturbation into the middle of model:
        # dimension (Width x Length x Depth) : 2640.0 m x 2640.0 m x 1.44 km
        for ispec  in range(0, nspec):
            for j in range(0, NGLLZ):
                for i in range(0, NGLLX):
                    # GLL point location (given in m: dimension 2640 m x 2640 x x 1440 m)
                    #iglob = ibool[i,j,ispec]
                    x = xstore[i,j,ispec]
                    z = zstore[i,j,ispec]

                    # vertical distance to center: at - 500 m depth
                    dist_v = np.sqrt( (CENTER_Z - z)*(CENTER_Z - z) )
                    # horizontal distance to center: at 1320 x 1320 m
                    dist_h = np.sqrt( (CENTER_X - x)*(CENTER_X - x) )
                    # Gaussian def:  values between [0,1]
                    pert_param[i,j,ispec] = np.exp( - (dist_h*dist_h) / sigma_h2 - (dist_v*dist_v) / sigma_v2 )

                    #debug
                    #print("debug: pert ",pert_param[i,j,ispec],x,z,dist_v,dist_h)

        # adds positive perturbation to model:
        if DO_PERTURB_RHO: rhostore = rhostore * (1.0 + percent * pert_param)
        if DO_PERTURB_VP:  vpstore  = vpstore * (1.0 + percent * pert_param)
        if DO_PERTURB_VS:  vsstore  = vsstore * (1.0 + percent * pert_param)

        print("  perturbed model velocity: rho min/max = ",rhostore.min(),rhostore.max())
        print("                            vp  min/max = ",vpstore.min(),vpstore.max())
        print("                            vs  min/max = ",vsstore.min(),vsstore.max())
        print("")

        # stores perturbed model files
        print("storing new files...")
        # rho
        filename = prname + "rho_gaussian.bin"
        # this didn't work:
        #np.asfortranarray(rhostore,'float32').tofile(filename)
        # nor:
        #out = rhostore.reshape(NGLLX,NGLLZ,nspec, order='F')
        #out.T.tofile(filename)
        # reading back these files in fortran code complains about having corrupted files.
        # thus, we use a custom routine here which puts back file markers around the arrays.
        hf.write_binary_file_custom_real_array(filename,rhostore)

        # vp
        filename = prname + "vp_gaussian.bin"
        hf.write_binary_file_custom_real_array(filename,vpstore)

        # vs
        filename = prname + "vs_gaussian.bin"
        hf.write_binary_file_custom_real_array(filename,vsstore)

        print("perturbed model files written to:")
        print("  ",prname + "rho_gaussian.bin")
        print("  ",prname + "vp_gaussian.bin")
        print("  ",prname + "vs_gaussian.bin")
        print("")

        # plot images
        if do_plot_figures:
            hf.plot_model_image(xstore,zstore,rhostore,prname + "rho_gaussian")
            hf.plot_model_image(xstore,zstore,vpstore,prname + "vp_gaussian")
            hf.plot_model_image(xstore,zstore,vsstore,prname + "vs_gaussian")


#
#------------------------------------------------------------------------------------------
#

def usage():
    print("Usage: ./model_add_Gaussian_perturbation.py param percent NPROC []")
    print("")
    print("    param   - model parameter, e.g., rho, vp or vs")
    print("    percent - perturbation strength (must be small enough (~1d-5) for F*dm=S(m+dm)-S(m) to be valid)")
    print("    NPROC   - number of MPI processes (for finding proc00000*_ model and kernel files)")
    print("    length  - (optional) wavelength of perturbation")
    sys.exit(1)

#
#------------------------------------------------------------------------------------------
#

if __name__ == '__main__':
    # gets arguments
    if len(sys.argv) < 4:
        usage()

    ## input parameters
    param = sys.argv[1]
    percent = float(sys.argv[2])
    NPROC = int(sys.argv[3])

    print("")
    print("model perturbation generation:")
    print("  perturbation = ",percent)
    print("  NPROC        = ",NPROC)
    print("")
    if len(sys.argv) == 5:
        length = float(sys.argv[4])
        print("  length       = ",length)

        # sets new wavelength for perturbation size
        wavelength = length
    print("")

    # initializes model perturbation flags
    DO_PERTURB_VP  =  False
    DO_PERTURB_VS  =  False
    DO_PERTURB_RHO =  False
    if param == "rho":
        DO_PERTURB_RHO = True
    elif param == "vp":
        DO_PERTURB_VP = True
    elif param == "vs":
        DO_PERTURB_VS = True
    elif param == "rhovp":
        DO_PERTURB_RHO = True
        DO_PERTURB_VP = True
    elif param == "rhovs":
        DO_PERTURB_RHO = True
        DO_PERTURB_VS = True
    elif param == "vpvs":
        DO_PERTURB_VP = True
        DO_PERTURB_VS = True
    elif param == "rhovpvs":
        DO_PERTURB_RHO = True
        DO_PERTURB_VP = True
        DO_PERTURB_VS = True

    model_add_Gaussian_perturbation(percent,NPROC,DO_PERTURB_RHO,DO_PERTURB_VP,DO_PERTURB_VS)

    print("")
    print("all done")
    print("")
