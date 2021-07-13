#!/bin/bash
module load imagemagick
module load python/3.8 mpi4py 
module load scipy-stack
module load StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 quast/5.0.2


save_one () {
    # echo 'Creating pngs'
    time python SnapPrints_scalar.py ${1} #${outpath} 
    rm -r $3/$1 #Remove if it exists so that it can replace 
    mv $2/$1 $3/
}

dapath=/project/def-ngrisoua/hannn 
outpath=$dapath/ifremer-pics_scalar 

# Prepare virtualenv
if [ ! -d "$SLURM_TMPDIR/env" ]; then
    virtualenv --no-download $SLURM_TMPDIR/env
    source $SLURM_TMPDIR/env/bin/activate
    pip3 install --no-index numpy netCDF4 joblib matplotlib  mpi4py #HW: add mpi4py, pip-->pip3(doesn't seem to make a difference)
else
    source $SLURM_TMPDIR/env/bin/activate
fi


cp $dapath/original-data/wp*.nc $SLURM_TMPDIR/. 
cp /home/hannn/pix2pix-for-swot/p2p_tf2_hw/SnapPrints_scalar.py $SLURM_TMPDIR/. 
#HW: delete $HOME
cd $SLURM_TMPDIR

ls -lh
placeholder='scalar' #needs to be consistent with cmap in master_scalar.bash
save_one $placeholder $(pwd) $outpath &  # The & allows mutliple parallel runs

exit
