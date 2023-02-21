#delete previous output
rm -rf OUTPUT*
rm -rf MODELS
rm -rf KERNELS
rm -rf DATA/*bin
rm -rf DATA/*jpg
rm -rf SEM
mv inversion.py inversion.tmp
mv helper_tyler.py helper_tyler.tmp
rm -f *.py
rm change*.pl
rm -rf bin
mv inversion.tmp inversion.py
mv helper_tyler.tmp helper_tyler.py

#restore previous state of DATA/Par_file
cp DATA/Par_file_ref DATA/Par_file
