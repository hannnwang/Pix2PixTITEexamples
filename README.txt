Codes to train and test TITE on a set of idealized data. Corresponding to paper "A deep learning approach to extract internal tides1scattered by geostrophic turbulence" to be submitted to Geophysical Research Letters. 

If you only want to take a quick look at how the architecture of the GAN looks like, you only need to read pix2pix_TF2_hw_train_scalar.py. Details of design considerations are included in the paper. If you are interested in running the program yourselves, see the instructions below. The instructions may be cumbersome, but most of them are just describing how to store or divide data so that the paths would match the paths specified in the programs. If you understand pix2pix_TF2_hw_train_scalar.py., you can well skip the descriptions below, creating your own paths for your data...

The instructions are written with the Graham cluster (https://docs.computecanada.ca/wiki/Graham) at Compute Canada in mind. Virtual environments are heavily used; if your clusters don't need that, the codes (i.e. anywhere you can see "virtualenv" or "$SLURM_TMPDIR")may be changed accordingly. 

a.)Prior to running, the .nc files downloaded from https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP2/HU58SG need to be pre-procceed. Follow these steps:
1. Save the downloaded .nc files into a folder titled "original-data" under a certain path. In "create-scalar-files.bash", change "datapath" into the path that contains "original-data".  
2. Save all the codes in this repo into a code folder. Replace "/home/hannn/pix2pix-for-swot/p2p_tf2_hw/" in "create-scalar-files.bash" with your code folder.
3. Run create-scalar-files.bash on your bash command line. Note that it will be usually too slow to run on log in mode. On Graham, I do: salloc --cpus-per-task=16 --mem=8G --time=00:30:00. It will take a few minutes to run.
4. Afterwards, under datapath/ifremer-pics_scalar, you should have a folder titled "scalar", under which you have three folders "ssh_lof", "ssh_cos" and "ssh_ins". They correspond to the QG component (not mentioned in paper), the cosine tidal component, and the total, of SSH respectively. All the SSH fields are saved as numpy arrays into .npz files. Each .npz file corersponds to one panel of a snapshot. 
In the naming, "wp50", "wp60","wp75","wp80" and "wp90" refer to simulations S1-5 respectively. "t1","t2" and "t3" means it belongs to the first 50 days, the second 50 days, or the third 50 days after the wave forcing is switched on. In the training and testing of pix2pix, we will only use t2 and t3. The numbers after t* denotes at which hour is it captured in t*. For example, t2-00072 means it's captured at the 53th day. "top","mid" and "bot" refer to the down-jet, mid-jet and up-jet panels respectively. "rec" means it's the rectangular  domain that avoids the sponge regions and the wave forcing centers.          
Note that these are just printouts; the inputs and outputs for pix2pix are not paired yet. 

Now, you can go on to run the pix2pix!
b.)For the "kitchensink" run, where the test images are randomly selected from all the S1-5 snapshots:
1.In "master_scalar_kitchensink.bash", you need to replace several folders in "master_scalar_kitchensink.bash". "tfenvdir" is the directory for your virtual environment. "picpath" should be the "datapath/ifremer-pics_scalar" path you created in step a/1-4 . "codedir" should be your code folder where you have copied all the codes in step a/2. "outdir" should be the folder where you store the outcomes. Note that it needs to be big enough.
Similarly, you need to replace the folder names and the slurm options (e.g. account, email address) in lanceur_scalar.slrm.
2. Now run "master_scalar_kitchensink.bash". This bash file can not be run on log-in mode either, due to the slightly cumbersome pairing of input and output images. I run it with salloc --cpus-per-task=16 --mem=16G --time=00:30:00.
Note the multiple options you can specify when running it. They mostly correspond to how you want to allocate the computational resources, as explained in the bash file. To get the results in paper, I run: 
bash master_scalar_kitchensink.bash -e 700 -m 40 -d 10 -s 113 -l 1000 -n 5
But if you are just trying the code out, you can just run 10 epochs first. For the kitchensink run, after 10 epochs, it can already start to generate something that looks plausible. If you are debugging, you may run it for just 10 epochs, and save checkpoints at every epoch. The default is to save the checkpoint every 10 epochs; to change it, you should modify "epochsave" in pix2pix_TF2_hw_train_scalar.py.
3. You should see a folder under your "outdir" titled with "setup" specified in "master_scalar_kitchensink.bash". Under it, test/. contains all pairs of testing instances, and train/. contains all pairs of training instances. training/training_checkpoints will contain all the checkpoitns you will save during the training. training/log/ will contain the loss function values during training. 

c.) After the jobs are complete, you can try the post-analysis programs in post_run_analysis. 
You'd first want to plot and see if the trained pix2pix is not generating garbage. This can be done at a low cost on your laptop. Follow these steps:
1. Create a local folder (e.g.on your laptop) for this run. For convenience, the name of the folder should be the same as "setup" that you have seen in step b/3 (i.e. the same name as your remote repo). 
Change the pattern of "runname" so that it matches the "setup" in "master_scalar_kitchensink.bash". The default in my code assumes you have run 700 epochs with nlayers set as 5 and so on. If you have changed the parameters in "setup", you need to change "runname" accordingly. 
Change "mpath" in restore_and_plot_ckpt_local.py to be the folder that contains this run. 
2. Look at the files saved in your outdir/training/training_checkpoints. The files formatted as "ckptX-Y.data-00000-of-00001" and "ckptX-Y.index" are the checkpoints. The larger X and Y are, the later the checkpoints are saved. X corresponds to the index of job run on the cluster, and Y corresponds to the index of the checkpoint. For example, when I have runned 700 epochs on the cluster with 10 jobs, then "ckpt10-70.index" means it's saved during the running of the 10th job, and it's the 70th checkpoint. If I have set "epochsave=10" in pix2pix_TF2_hw_train_scalar.py, then the 70th checkpoint correspond to the 700th epoch. 
Decide which checkpoint you want to apply. (Any checkpoint saved after the 10th epoch should work if you just want see whether or not pix2pix is learning garbage.)
For example say you want to look at the 600th epoch and the corresponding checkpoint files are "ckpt9-60.data-00000-of-00001" and "ckpt9-60.index"
3. Create the folder "training/training_checkpoints" under mpath. Copy the above two checkpoint files, as well as the file titled "checkpoint" under it. This may take a while as the files are ~600MB.  
4. Change "ckptname" in restore_and_plot_ckpt_local.py into the "ckptX-Y" that is consistent with the chekpoints you just copied.
Check if the parameters BATCH_SIZE, N_LAYERS and so on in restore_and_plot_ckpt_local.py are the same as the ones you have used.
5. Create a folder titled "test" under mpath. Choose a few testing instances you want to look at from your remote folder outdir/test/.  The naming of these numpy files are similar as the ones described in step a/4. 
You don't have to copy all the test files; for a first test, you are recommend to only copy about 3 files. 
6. Run restore_and_plot_ckpt_local.py. By default, it should a figure (4 panels) for each test instance under your test folder. If you don't want so many figures, or only want specific test instances, you can modify the codes yourself.  

If you would like to try more analytics on the results, you can apply most parts of the codes above to restore the checkpoints and generate images, and write your own codes to compute correlations and so on. I did not upload my codes on those because there are too many of them, and I believe the computations are all standard, and described in details in the paper (or supporting information.)

d.) For the ES1-5 runs, the steps are similar, but running "master_scalar.bash" instead. I'm yet to caption it.
