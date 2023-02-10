#delete previous output
rm -rf OUTPUT*
rm -rf MODELS
rm -rf KERNELS
rm -rf DATA/*bin
rm -rf DATA/*jpg
rm -rf SEM
rm -f *.py
rm change*.pl
rm -rf bin

#restore previous state of DATA/Par_file
cp DATA/Par_file_orig DATA/Par_file
