#!/bin/bash
#
# script runs mesher and solver (in serial)
# using this example setup
#
########################### synthetics ################################

iter_no=$1

#make sure we have right Par_file version
cp DATA/Par_file_inter DATA/Par_file
if [[ $iter_no == 1 ]]
then
    rm misfitx.log
    rm misfitz.log
fi

#perform redundant grepping that is already done in setup.sh
NPROC=`grep '^NPROC ' DATA/Par_file | cut -d = -f 2 | cut -d \# -f 1 | tr -d ' '`
NSTEP=`grep '^NSTEP ' DATA/Par_file | cut -d = -f 2 | cut -d \# -f 1 | tr -d ' '`
DT=`grep '^DT ' DATA/Par_file | cut -d = -f 2 | cut -d \# -f 1 | tr -d ' '`
SIM_TYPE=2

##
## synthetics simulation
##
echo
echo "running synthetics forward simulation (with saving forward wavefield)"
echo
./change_simulation_type.pl -F

# Par_file using GLL model
sed -i '' "s/^MODEL .*=.*/MODEL = gll/" DATA/Par_file
sed -i '' "s/^SAVE_MODEL .*=.*/SAVE_MODEL = .false./" DATA/Par_file

./run_this_example.sh > output.log

# checks exit code
if [[ $? -ne 0 ]]; then exit 1; fi

echo
mv -v output.log OUTPUT_FILES/output.forward.log

# backup copy
rm -rf OUTPUT_FILES.syn.forward/
cp -rp OUTPUT_FILES OUTPUT_FILES.syn.forward

# initial model
mv -v OUTPUT_FILES/*.su SEM/syn/
cp -v DATA/*rho.bin MODELS/initial_model/
cp -v DATA/*vp.bin  MODELS/initial_model/
cp -v DATA/*vs.bin  MODELS/initial_model/

cp -v DATA/*NSPEC_ibool.bin  MODELS/initial_model/
cp -v DATA/*x.bin  MODELS/initial_model/
cp -v DATA/*z.bin  MODELS/initial_model/

########################### adj sources ################################
## adjoint sources
echo
echo "creating adjoint sources..."

# x-component
if [ -e OUTPUT_FILES.syn.forward/Ux_file_single_d.su ]; then
  syn=OUTPUT_FILES.syn.forward/Ux_file_single_d.su
  dat=OUTPUT_FILES.dat.forward/Ux_file_single_d.su
  mode=$2
  MISFIT_LOG=misfitx.log
  echo "> ./adj_seismogram.py $syn $dat $mode $MISFIT_LOG"
  echo
  # adjoint source f^adj = (s - d)
  ./adj_seismogram.py $syn $dat $mode $MISFIT_LOG
  # checks exit code
  if [[ $? -ne 0 ]]; then exit 1; fi
fi

# y-component
if [ -e OUTPUT_FILES.syn.forwardUy_file_single_d.su ]; then
  echo "HITTING Y COMPONENT!!!"
  sleep 10
  syn=OUTPUT_FILES.syn.forward/Uy_file_single_d.su
  dat=OUTPUT_FILES.dat.forward/Uy_file_single_d.su
  echo "> ./adj_seismogram.py $syn $dat"
  echo
  # adjoint source f^adj = (s - d)
  ./adj_seismogram.py $syn $dat
  # checks exit code
  if [[ $? -ne 0 ]]; then exit 1; fi
fi

# z-component
if [ -e OUTPUT_FILES.syn.forward/Uz_file_single_d.su ]; then
  syn=OUTPUT_FILES.syn.forward/Uz_file_single_d.su
  dat=OUTPUT_FILES.dat.forward/Uz_file_single_d.su
  mode=$2
  MISFIT_LOG=misfitz.log
  echo "> ./adj_seismogram.py $syn $dat $mode $MISFIT_LOG"
  # adjoint source f^adj = (s - d)
  ./adj_seismogram.py $syn $dat $mode $MISFIT_LOG
  # checks exit code
  if [[ $? -ne 0 ]]; then exit 1; fi
fi

########################### kernel ################################

## kernel simulation
echo
echo "running kernel simulation"
echo
./change_simulation_type.pl -b

# In principle we do not need rerun xmeshfem2D in the adjoint run.
# However, in the current structure of the code, the xspecfem2D program can not detect the
# the parameter change in Par_file. Since in the adjoint run we change the SIMULATION_TYPE and the save_forward

./run_this_example.sh noclean > output.log

# checks exit code
if [[ $? -ne 0 ]]; then exit 1; fi

echo
mv -v output.log OUTPUT_FILES/output.kernel.log

# backup
rm -rf OUTPUT_FILES.syn.adjoint/
cp -rp OUTPUT_FILES OUTPUT_FILES.syn.adjoint

# kernels
cp -vp OUTPUT_FILES/output.kernel.log KERNELS/
cp -vp OUTPUT_FILES/*_kernel.* KERNELS/


########################### model update ################################

echo
echo "model update"
echo

# takes absolute value of percent
update_percent=$(cat setup.sh | grep "perturb_percent=" | sed 's/perturb_percent=//' | sed 's/-//')

echo "> ./model_update.py $NPROC $SIM_TYPE $update_percent"
echo
./model_update.py $NPROC $SIM_TYPE $update_percent

# checks exit code
if [[ $? -ne 0 ]]; then exit 1; fi

# replaces model files with perturbed ones
for (( iproc = 0; iproc < $NPROC; iproc++ )); do
  rank=`printf "%06i\n" $iproc`
  cp -v DATA/proc${rank}_rho_new.bin DATA/proc${rank}_rho.bin
  cp -v DATA/proc${rank}_vp_new.bin DATA/proc${rank}_vp.bin
  cp -v DATA/proc${rank}_vs_new.bin DATA/proc${rank}_vs.bin
  if [[ $? -ne 0 ]]; then exit 1; fi
done

########################### postprocessing ################################
if [[ $(($iter_no > 1)) ]]
then
  ./kernel_evaluation_postprocessing.py $NSTEP $DT $NPROC $SIM_TYPE
fi

########################### final forward ################################

# ## forward simulation
# echo
# echo "running forward simulation (updated model)"
# echo
# ./change_simulation_type.pl -f

# # In principle we do not need rerun xmeshfem2D in the adjoint run.
# # However, in the current structure of the code, the xspecfem2D program can not detect the
# # the parameter change in Par_file. Since in the adjoint run we change the SIMULATION_TYPE and the save_forward

# ./run_this_example.sh > output.log
# # checks exit code
# if [[ $? -ne 0 ]]; then exit 1; fi

# echo
# mv -v output.log OUTPUT_FILES/output.log

# # backup
# rm -rf OUTPUT_FILES.syn.updated
# cp -rp OUTPUT_FILES OUTPUT_FILES.syn.updated


# ########################### kernel ################################

# echo
# echo "postprocessing..."
# echo "> ./kernel_evaluation_postprocessing.py $NSTEP $DT $NPROC $SIM_TYPE"
# echo
# ./kernel_evaluation_postprocessing.py $NSTEP $DT $NPROC $SIM_TYPE

# # checks exit code
# if [[ $? -ne 0 ]]; then exit 1; fi

# echo
# echo "done: `date`"
# echo
