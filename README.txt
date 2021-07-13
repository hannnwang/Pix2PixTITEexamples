The instructions are written with the Graham cluster (https://docs.computecanada.ca/wiki/Graham) at Compute Canada in mind. Virtual environments are heavily used; if your clusters don't need that, the codes (i.e. anywhere you can see "virtualenv" or "$SLURM_TMPDIR")may be changed accordingly. 

a.)Prior to running, the .nc files downloaded from https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP2/HU58SG need to be pre-procceed. Following these steps:
1. Save the downloaded .nc files into a folder titled "original-data" under a certain path. In "create-scalar-files.bash", change "datapath" into the path that contains "original-data".  
2. Save all the codes in this repo into a code folder. Replace "/home/hannn/pix2pix-for-swot/p2p_tf2_hw/" in "create-scalar-files.bash" with your code folder.
3. Run create-scalar-files.bash on your bash command line. Note that it will be usually too slow to run on log in mode. On Graham, I do: salloc --cpus-per-task=16 --mem=8G --time=00:30:00. It will take a few minutes to run.
4. Afterwards, under datapath/ifremer-pics_scalar, you should have a folder titled "scalar", under which you have three folders "ssh_lof", "ssh_cos" and "ssh_ins". They correspond to the QG component (not mentioned in paper), the cosine tidal component, and the total, of SSH respectively. All the SSH fields are saved as numpy arrays into .npz files. Each .npz file corersponds to one panel of a snapshot. 
In the naming, "wp50", "wp60","wp75","wp80" and "wp90" refer to simulations S1-5 respectively. "t1","t2" and "t3" means it belongs to the first 50 days, the second 50 days, or the third 50 days after the wave forcing is switched on. In the training and testing of pix2pix, we will only use t2 and t3. The numbers after t* denotes at which hour is it captured in t*. For example, t2-00072 means it's captured at the 53th day. "top","mid" and "bot" refer to the down-jet, mid-jet and up-jet panels respectively. "rec" means it's the rectangular  domain that avoids the sponge regions and the wave forcing centers.          
Note that these are just printouts; the inputs and outputs for pix2pix are not paired yet. 

Now, you can go on to run the pix2pix!
b.)For the "kitchensink" run, where the test images are randomly selected from all the S1-5 snapshots:
1.In "master_scalar_kitchensink.bash", you need to replace several folders in "master_scalar_kitchensink.bash". "tfenvdir" is the directory for your virtual environment. "picpath" should be the "datapath/ifremer-pics_scalar" path you created in step a/1-4 . "codedir" should be your code folder where you have copied all the codes in step a/2. "outdir" should be the folder where you store the outcomes. Note that it needs to be big enough.
2. Now run "master_scalar_kitchensink.bash". This bash file can not be run on log-in mode either, due to the slightly cumbersome pairing of input and output images. I run it with salloc --cpus-per-task=16 --mem=16G --time=00:30:00.
Note the multiple options you can specify when running it. They mostly correspond to how you want to allocate the computational resources, as explained in the bash file. To get the results in paper, I run: 
bash master_scalar_kitchensink.bash -e 700 -m 40 -d 10 -s 113 -l 1000 -n 5
3. You should see a folder under your "outdir" titled with "setup" specified in "master_scalar_kitchensink.bash". Under it, test/. contains all pairs of testing instances, and train/. contains all pairs of training instances. training/training_checkpoints will contain all the checkpoitns you will save during the training. training/log/ will contain the loss function values during training. 
4. After the jobs are complete, you can try the post-analysis programs in post_run_analysis (I am still captioning them). 

