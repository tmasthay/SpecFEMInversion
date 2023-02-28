#!/bin/bash
#
# script runs mesher and solver (in serial)
# using this example setup
#
echo "running example: `date`"
currentdir=`pwd`

cd $currentdir

rm -f adj_seismogram.py
ln -s ../adj_seismogram.py
rm -f model_add_Gaussian_perturbation.py
ln -s ../model_add_Gaussian_perturbation.py
rm -f model_update.py
ln -s ../model_update.py
rm -f kernel_evaluation_postprocessing.py
ln -s ../kernel_evaluation_postprocessing.py
rm -f helper_functions.py
ln -s ../helper_functions.py
rm -f helper_tyler.py
ln -s ../helper_tyler.py

rm -f wasserstein.py
ln -s ../wasserstein.py

rm -f change_simulation_type.pl
ln -s $SPEC/utils/change_simulation_type.pl

rm -f create_STATIONS_file.py
ln -s ../create_STATIONS_file.py

# Make sure to restore previous state
cp DATA/Par_file_ref DATA/Par_file
./helper_tyler.py 2

##############################################
## Elastic benchmark

# Simulation type 1 == acoustic / 2 == elastic P-SV / 3 == elastic SH / 4 == coupled acoustic-elastic
SIM_TYPE=2

# perturbation model parameter rho/vp/vs (e.g. "rho" or "vp" or "rhovp")
perturb_param="vp"

# perturbation (should be small enough for approximating S(m - m0) ~ S(m) - S(m0)
perturb_percent=-50.0

# number of stations along x/z lines
nlinesx=1
nlinesz=1

##############################################

echo
echo "setup:"
echo "  SIM_TYPE                : $SIM_TYPE     (1 == acoustic / 2 == elastic P-SV / 3 == elastic SH / 4 == coupled acoustic-elastic)"
echo "  perturbation parameter  : $perturb_param"
echo "  perturbation percent    : $perturb_percent"
echo "  number of stations/lines: $nlinesx $nlinesz"
echo

# SH-wave simulation setup
if [ "$SIM_TYPE" == "3" ]; then
  # source
  sed -i '' "s/^source_type .*/source_type  = 1/" $(pwd)/DATA/SOURCE
  sed -i '' "s/^time_function_type .*/time_function_type  = 1/" $(pwd)DATA/SOURCE
  sed -i '' "s/^factor .*/factor  = 1.d10/" $(pwd)/DATA/SOURCE
  # Par_file
  sed -i '' "s/^P_SV .*/P_SV  = .false./" DATA/Par_file
  sed -i '' "s/^NSTEP .*/NSTEP  = 900/" DATA/Par_file
  # checks param selection for SH-wave simulations
  if [ "$perturb_param" == "vp" ]; then
    # switch to vs perturbations instead
    echo "SH-wave simulation: switching perturbation parameter vp to vs"
    echo
    perturb_param="vs"
  fi
fi

# gets Par_file parameters
# Get the number of processors
NPROC=`grep '^NPROC ' DATA/Par_file | cut -d = -f 2 | cut -d \# -f 1 | tr -d ' '`
NSTEP=`grep '^NSTEP ' DATA/Par_file | cut -d = -f 2 | cut -d \# -f 1 | tr -d ' '`
DT=`grep '^DT ' DATA/Par_file | cut -d = -f 2 | cut -d \# -f 1 | tr -d ' '`

echo "Par_file parameters:"
echo "  NPROC = $NPROC"
echo "  NSTEP = $NSTEP"
echo "  DT    = $DT"
echo

# create and compile all setup
do_setup=1

##############################################

if [ "$do_setup" == "1" ]; then
echo
echo "setting up example..."
echo

# cleans files
rm -rf DATA/*.bin

mkdir -p OUTPUT_FILES
rm -rf OUTPUT_FILES/*
mkdir -p OUTPUT_FILES/

mkdir -p SEM
rm -rf SEM/*
mkdir -p SEM/dat SEM/syn

mkdir -p MODELS
rm -rf MODELS/*

mkdir -p MODELS/initial_model MODELS/target_model

mkdir -p KERNELS
rm -rf KERNELS/*

# creates STATIONS file
./create_STATIONS_file.py $nlinesx $nlinesz

# checks exit code
if [[ $? -ne 0 ]]; then exit 1; fi

fi

########################### data ########################################

##
## data simulation
##

## forward simulation
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
rm -rf OUTPUT_FILES.dat.forward
cp -rp OUTPUT_FILES OUTPUT_FILES.dat.forward

cp -v OUTPUT_FILES/*.su SEM/dat/

# target model
cp -v DATA/*rho.bin MODELS/target_model/
cp -v DATA/*vp.bin  MODELS/target_model/
cp -v DATA/*vs.bin  MODELS/target_model/

cp -v DATA/*NSPEC_ibool.bin  MODELS/target_model/
cp -v DATA/*x.bin  MODELS/target_model/
cp -v DATA/*z.bin  MODELS/target_model/

########################### model perturbation ################################

echo
echo "setting up perturbed model..."
echo "> ./model_add_Gaussian_perturbation.py $perturb_param $perturb_percent $NPROC "
echo
#./model_add_Gaussian_perturbation.py $perturb_param $perturb_percent $NPROC
./helper_tyler.py 6 $perturb_percent

# checks exit code
if [[ $? -ne 0 ]]; then exit 1; fi

# replaces model files with perturbed ones
for (( iproc = 0; iproc < $NPROC; iproc++ )); do
  rank=`printf "%06i\n" $iproc`
  cp -v DATA/proc${rank}_rho_gaussian.bin DATA/proc${rank}_rho.bin
  cp -v DATA/proc${rank}_vp_gaussian.bin DATA/proc${rank}_vp.bin
  cp -v DATA/proc${rank}_vs_gaussian.bin DATA/proc${rank}_vs.bin
  if [[ $? -ne 0 ]]; then exit 1; fi
done

cp DATA/Par_file DATA/Par_file_inter
