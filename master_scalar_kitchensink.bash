#!/bin/bash

function usage () {
    cat <<EOUSAGE
$(basename $0) :hicwev
master.bash should be used with arguments. Complete list:
    -h   help
    -i   Colormap for the raw (instantaneous) SSH (you shouldn't change this.)
    -c   Colormap for the cosine-fitted (wave) SSH (you shouldn't change this.)
    -e   Total number of epochs for training
    -m   Memory for each job
    -d   How many jobs to run in order to cover the total number of epochs. Ideally mod(e/d)=0 
    -s   How many seconds each epoch takes  
	-l   Parameter Lambda controling the L1 loss as shown in paper
	-n   number of layers in the discriminator. Setting n=3 corresponds to the "patchGAN" with a patch size of 75, as shown in Isola et al. Setting n=5 corresponds to the "imageGAN", from which the discriminator treats the whole image at once. 
	
EOUSAGE
}

#Can not be run on log-in mode; I'm running it on interactive sessions, with 16GB memory 

echo 'Creating and submitting a new experiment.'
echo 'See master.bash -h for help on arguments.' 

# All arguments are optional, here we assign them
while getopts :hi:c:e:m:d:s:l:n: opt; do 
    case $opt in  
        h) usage; exit 1;;
        i) cm_ins=${OPTARG};;
        c) cm_cos=${OPTARG};;
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

cm_ins=${cm_ins:-scalar} #Default: scalar numpy files 
cm_cos=${cm_cos:-scalar} #Default: scalar numpy files
n_epochs=${n_epochs:-1000} #Default: 1000 epochs in total(probably too much)
n_mem=${n_mem:-70} #Default: 70 GB memory request for each job (probably too much)
divide=${divide:-1} ##Default: job division is 1. 
epoch_sec=${epoch_sec:-75}
LAMBDA=${LAMBDA:-100} #Default: 100, as explained in paper.
n_layers=${n_layers:-3}

cat <<EOF

The experiment has the following parameters:
    n_epochs = ${n_epochs}
	LAMBDA = ${LAMBDA}
    
EOF
echo 'n_mem:' $n_mem
echo 'epoch_sec:' $epoch_sec
nepoch_eachjob=$((n_epochs / divide)) #How many epochs each job is going to run. Note: can only divide integers  
divn=$(( 3600 / epoch_sec ))
n_hours=$(( (nepoch_eachjob + 10)/ divn )) #Add 10 epochs' time for saving etc. Note that shell divisions are all integar-based. 
n_hours=$(( n_hours + 1 ))  # +1 to round up  (combined with previous commands, this is equivalent to ceil())

printf -v n_hours "%02d" $n_hours  # padding with zero (-v means printing to variable)
echo 'n_hours for each job:' $n_hours 
echo 'total epochs:' $n_epochs
echo 'epoch number in each job:' $nepoch_eachjob 

# Prepare virtualenv
module load python/3.8 
    printf "\n\n :X:X:X TensorfFlow v2 :X:X:X \n\n"
    tfenvdir=/home/hannn/pix2pix-for-swot/tfV2envre
    process_script=process_scalar.py  
	p2p_script_train=pix2pix_TF2_hw_train_scalar.py

if [ ! -d "$tfenvdir" ]; then  
    # we need tensorflow for the process.py script
    virtualenv --no-download $tfenvdir
    source $tfenvdir/bin/activate
    pip install --no-index tensorflow_cpu 
else
    source $tfenvdir/bin/activate
fi

# # Prepare data
setup=lambda_${LAMBDA}_nlayers_${n_layers}_kitchen_sink_${n_epochs}epochs_w_rotflipud

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

#Matching the input and output images and patch them side-by-side. 
echo 'Combining colormaps ' $cm_ins ' and ' $cm_cos

python $codedir/tools/${process_script} \
       --input_dir $picpath/$cm_ins/ssh_ins \
       --b_dir $picpath/$cm_cos/ssh_cos \
       --operation combine \
       --output_dir $outdir/combined

deactivate  # deactivate the tensorflow environment

echo 'Done: process_script. Now dividing into test and train folders'


cd $outdir
cp $codedir/lanceur_scalar.slrm .
cp -r $codedir $outdir/. #added by hw 
#cp $codedir/$p2p_script_train_first $outdir/.
cp $codedir/$p2p_script_train $outdir/.
#cp -r $codedir/tools/ $outdir/.

mkdir train test  # careful again if checkpointing or testing

mkdir testing #hw added to store tested images

#combined/. originally contains all  pairs of panels. Then, we move all the t2 and t3 files to train/. From train/., we randomly select 20%, and move them to test/.

mv combined/wp*t2*-3_*.npy train/.
mv combined/wp*t3*-3_*.npy train/.

nall=$(ls train | wc -l)
ntrain=$(($nall/5)) #Randomly take 20%
testfiles=$(ls train | shuf -n $ntrain)
for filename in $testfiles
do 
	mv train/$filename test/.
done

#combined/. is now useless.

tar cf data.tar train test  # archive for $SLURM_TMPDIR 


echo 'submitting SLURM jobs'

ed -s "lanceur_scalar.slrm" <<< $'g/XXXX/s/XXXX/'${cm_ins}$'/g\nw\nq' #Replacing XXXX with cm_ins
ed -s "lanceur_scalar.slrm" <<< $'g/YYYY/s/YYYY/'${cm_cos}$'/g\nw\nq'
ed -s "lanceur_scalar.slrm" <<< $'g/XYXY/s/XYXY/'${nepoch_eachjob}$'/g\nw\nq'
ed -s "lanceur_scalar.slrm" <<< $'g/ZXZX/s/ZXZX/'${n_hours}$'/g\nw\nq'
ed -s "lanceur_scalar.slrm" <<< $'g/MEMO/s/MEMO/'${n_mem}$'/g\nw\nq' 
ed -s "lanceur_scalar.slrm" <<< $'g/ZZZZ/s/ZZZZ/'${p2p_script_train}$'/g\nw\nq'
ed -s "lanceur_scalar.slrm" <<< $'g/NJOBS/s/NJOBS/'${divide}$'/g\nw\nq' 
ed -s "lanceur_scalar.slrm" <<< $'g/LLLL/s/LLLL/'${LAMBDA}$'/g\nw\nq'
ed -s "lanceur_scalar.slrm" <<< $'g/NLNL/s/NLNL/'${n_layers}$'/g\nw\nq'

sbatch --job-name=$setup lanceur_scalar.slrm

exit
