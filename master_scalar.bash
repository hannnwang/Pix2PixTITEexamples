#!/bin/bash

function usage () {
    cat <<EOUSAGE
$(basename $0) :hicwev
master.bash should be used with arguments. Complete list:
    -h   help
    -i   Matplotlib colormap for the raw (instantaneous) SSH (needs to be run by create_pics_ssavehw.bash)
    -c   Matplotlib colormap for the cosine-fitted (wave) SSH (needs to be run by create_pics_ssavehw.bash)
    -w   Series to test the model on (format: wp*8_[ab])
	-t   Series to train the model on (format:wp*8_[cdefg])(Note: can also contain test images, thanks to the order we move things.)
    -e   Total number of epochs for training
    -m   Memory for each job
    -d   How many jobs to span the total number of epochs. Ideally mod(e/d)=0 
    -s   How many seconds each epoch takes  
	-l   Parameter Lambda controling the L1 loss 
	
EOUSAGE
}

#Can not be run on log-in mode; I'm running it on interactive sessions, with 16GB memory 
#hw: changed from Nico's version, so that this runs exclusively for TensorfFlow 2. Supressing the -v, tfv options related stuff, and replace codes with tf2 counterparts. 

echo 'Creating and submitting a new experiment.'
echo 'See master.bash -h for help on arguments.' 

# All arguments are optional, here we assign them
while getopts :hi:c:w:t:e:m:d:s:l:n: opt; do #****"hi" may be wrong 
    case $opt in  #So probably the syntax is : ./master -i gray -c gray
        h) usage; exit 1;;
        i) cm_ins=${OPTARG};;
        c) cm_cos=${OPTARG};;
        w) wp_test=${OPTARG};;
		t) wp_train=${OPTARG};;
        e) n_epochs=${OPTARG};;
	m) n_mem=${OPTARG};;
        d) divide=${OPTARG};;
		s) epoch_sec=${OPTARG};;
		l) LAMBDA=${OPTARG};;
		n) n_layers=${OPTARG};;
		
        \?) echo "Invalid option -${OPTARG}" >&2
            usage; exit 2;;
    esac
done
#These are assigning the default values in case the command line has no input
cm_ins=${cm_ins:-scalar} 
cm_cos=${cm_cos:-scalar}
wp_test=${wp_test:-wp50*-3}
wp_train=${wp_train:-wp*-3}
n_epochs=${n_epochs:-700}
n_mem=${n_mem:-30}
divide=${divide:-7}
epoch_sec=${epoch_sec:-90} 
LAMBDA=${LAMBDA:-1000}
n_layers=${n_layers:-3}

cat <<EOF

The experiment has the following parameters:
    wp_test = ${wp_test}
	wp_train = ${wp_train}
    n_epochs = ${n_epochs}
	LAMBDA = ${LAMBDA}
	NLAYERS = ${n_layers}
    
EOF
echo 'n_mem:' $n_mem
echo 'epoch_sec:' $epoch_sec
nepoch_eachjob=$((n_epochs / divide)) #Note: bash can only divide integers  
divn=$(( 3600 / epoch_sec ))
n_hours=$(( (nepoch_eachjob + 10)/ divn )) # for each sub-job. Note that shell divisions are all integar-based. 
#n_hours=$(( n_hours + 1 ))  # +1 to round up  (combined with previous commands, this is equivalent to ceil())

printf -v n_hours "%02d" $n_hours  # padding with zero (-v means printing to variable)
# Graham does not accept jobs longer than 24 hrs. Unlikely so not failsafing
echo 'n_hours for each job:' $n_hours 
echo 'total epochs:' $n_epochs
echo 'epoch number in each job:' $nepoch_eachjob 


# Prepare virtualenv
module load python/3.8 #Changed to 3.8
#if [ "$tfv" == 2 ]; then
    printf "\n\n :X:X:X TensorfFlow v2 :X:X:X \n\n"
    tfenvdir=/home/hannn/pix2pix-for-swot/tfV2envre
    process_script=process_scalar.py  
    #p2p_script_train_first=pix2pix_TF2_hw_train_first.py
	p2p_script_train=pix2pix_TF2_hw_train_scalar.py
#else
#    tfenvdir=/home/hannn/pix2pix-for-swot/tfV1env
#    process_script=process.py
#    p2p_script=pix2pix.py
#fi

if [ ! -d "$tfenvdir" ]; then  
    # we need tensorflow for the process.py script
    virtualenv --no-download $tfenvdir
    source $tfenvdir/bin/activate
    pip install --no-index tensorflow_cpu #***check version 
#   pip install --no-index tensorflow_cpu==1.14.1
else
    source $tfenvdir/bin/activate
fi

# # Prepare data
setup=lambda_${LAMBDA}_nlayers_${n_layers}_testing-${wp_test}_${n_epochs}epochs_w_rotflipud #Change the job name to this for debugging memory leak

#Delete some special characters:
setup=$(echo $setup | tr '][*-' '_')  #Important for a proper function of glob. in python etc. 
setup=$(echo $setup | tr -s '_')
  
echo "Name of this experiment: ${setup}"

picpath=/home/hannn/projects/def-ngrisoua/hannn/ifremer-pics_scalar
codedir=/home/hannn/pix2pix-for-swot/p2p_tf2_hw
outdir=/scratch/hannn/pix2pix-for-swot/$setup  
if [ ! -d "$outdir" ]; then #-d is a operator to test if the given directory exists or not.
    mkdir $outdir
else
    rm -r $outdir  # careful if checkpointing
	mkdir $outdir
fi

echo ' '

#------------------------------------Caption below if it's already done.
echo 'Combining colormaps ' $cm_ins ' and ' $cm_cos


python $codedir/tools/${process_script} \
       --input_dir $picpath/$cm_ins/ssh_ins \
       --b_dir $picpath/$cm_cos/ssh_cos \
       --operation combine \
       --output_dir $outdir/combined
#------------------------------------Caption above if it's already done.

deactivate  # deactivate the tensorflow environment

echo 'Done: process_script. Now dividing into test and train folders'


cd $outdir
cp $codedir/lanceur_scalar.slrm .
cp -r $codedir $outdir/. #added by hw 
#cp $codedir/$p2p_script_train_first $outdir/.
cp $codedir/$p2p_script_train $outdir/.
#cp -r $codedir/tools/ $outdir/.

mkdir train test  # careful again if checkpointing or testing
mkdir validation #newly added on 2021/05/29

#Note: since we move the test images first, the -t (training images) in the command line can contain test images too.
mv combined/${wp_test}*.npy test/.  #Probably no need to shuffle #Moving the test data to test/.
mv combined/${wp_train}*.npy train/.  #hw added so that we can specify what to train

#randomly select 20 percent of the snapshots in training data for validation:
#Note: we just randomly select the SNAPSHOTS, and include all the top/mid/bot panels in those snapshots.
echo "WARNING: the way we choose validation data only works for panels denoted by top/mid/bot!"
nalltop=$(ls train/*top* | wc -l) #number of filenames containing "top"
nvalidationtop=$(($nalltop/5))
validationfilestop=$(ls train/*top* | shuf -n $nvalidationtop)
for filename in $validationfilestop
do 
	validationfilestr=${filename%top.npy} #Remove "top.npy" from the back of the string. 
	#echo $validationfilestr
	mv ${validationfilestr}*.npy validation/.	
done


tar cf data.tar train test validation  # archive for $SLURM_TMPDIR # useful in pix2pix.py


echo 'submitting SLURM jobs'

ed -s "lanceur_scalar.slrm" <<< $'g/XXXX/s/XXXX/'${cm_ins}$'/g\nw\nq' #Replacing XXXX with cm_ins
ed -s "lanceur_scalar.slrm" <<< $'g/YYYY/s/YYYY/'${cm_cos}$'/g\nw\nq'
ed -s "lanceur_scalar.slrm" <<< $'g/XYXY/s/XYXY/'${nepoch_eachjob}$'/g\nw\nq'
ed -s "lanceur_scalar.slrm" <<< $'g/ZXZX/s/ZXZX/'${n_hours}$'/g\nw\nq'
ed -s "lanceur_scalar.slrm" <<< $'g/MEMO/s/MEMO/'${n_mem}$'/g\nw\nq' #added for memory control
ed -s "lanceur_scalar.slrm" <<< $'g/ZZZZ/s/ZZZZ/'${p2p_script_train}$'/g\nw\nq'
ed -s "lanceur_scalar.slrm" <<< $'g/NJOBS/s/NJOBS/'${divide}$'/g\nw\nq' 
ed -s "lanceur_scalar.slrm" <<< $'g/LLLL/s/LLLL/'${LAMBDA}$'/g\nw\nq'
ed -s "lanceur_scalar.slrm" <<< $'g/NLNL/s/NLNL/'${n_layers}$'/g\nw\nq'

sbatch --job-name=$setup lanceur_scalar.slrm


exit
