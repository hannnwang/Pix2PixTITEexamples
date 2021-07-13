 #prints the scalar values directly as numpy arrays.
 # The file has been written by different people, and sorry about the various coding styles.
from sys import argv  
import os  
import shutil  
import numpy as np
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt  
from joblib import Parallel, delayed
import multiprocessing
from netCDF4 import Dataset  


def create_datasets_list(abs_path_of_top):
    """ From the top directory (may be the local one actually), this function
    returns a list of all eligible absolute paths containing the appropriate
    files
    The operator would give the directory containing the jet_XXX directories
    IN: abs_path_of_top (string). Empty (default) compute abspath of parent
    OUT: list of absolute paths (['ap0', 'ap1', ...])
    """

    list_of_ncs = []
    for root, _, files in os.walk(abs_path_of_top):# os.walk returns a tuple with the current_folder, a list of sub_folders, and a list of files in the current_folder
        for ff in files:
            if ff.startswith('file_ave_'):  # should be actually useless after HW's modifications.
                list_of_ncs.append(os.path.join(root, ff)) 
            if ff.startswith('wp'):  # for publishable-data
                list_of_ncs.append(os.path.join(root, ff))

    if not list_of_ncs:
        raise NameError('I found no appropriate datasets')

    return sorted(list_of_ncs)


def find_global_minmax(list_of_netcdfs, list_of_variables):
    first_iteration = True  # nothing to compare on the first iteration
    for ds in list_of_netcdfs:
        fid = Dataset(ds, mode='r')
        for vv in list_of_variables:
            to_plot = fid.variables[vv][:]
            glob_min_loc, glob_max_loc = np.amin(to_plot), np.amax(to_plot)
            if first_iteration:
                glob_min, glob_max = glob_min_loc*1., glob_max_loc*1.
                first_iteration = False
            else:
                glob_min = min([glob_min, glob_min_loc])
                glob_max = max([glob_max, glob_max_loc])

        fid.close()

    return glob_min, glob_max

def process_one_dataset(ds, list_to_plt, vm, vM, FigPth, npix, cmap):
    """ From the absolute path of an .nc file, print all of the snapshots"""
    fid = Dataset(ds, mode='r')

    print("Processing "+str(ds))

    iwp = ds.find('wp')  # index of the wp sub-string. What follows is wp
    wpid = ds[iwp:-3]
    
    # %% Let's plot a couple of things ---------------------------------------|
    fig, ax = plt.subplots()

    for ll in list_to_plt:
        # different variables have slightly different coordinates, which is not a problem for now since we are only printing ssh. But if we are printing e.g. velocity fields too, this would have to be changed.
        coords = fid.variables[ll].coordinates.split()
        # the above yields a list, each element is a coordinate: t, x, y resp.
        # the below works because each coordinate above is also a variable
        time_array = fid.variables[coords[0]][:]
        xlon = fid.variables[coords[1]][:]
        ylat = fid.variables[coords[2]][:]
        ny, nx = xlon.shape
    
        if squaresYN: 
            slc_y = {'3_bot': slice(104, 104+nx),  #up-jet panel
           '3_mid': slice(104+nx//2, 104+nx//2+nx), #mid-jet panel
           '3_top': slice(104+nx, 104+2*nx), #down-jet panel
           '1_rec': slice(104, 104+2*nx), #The rectangular domain that avoids the sponge regionsand the wave forcing centers.          
           } 
    
            slc_x={'3_bot': slice(0, nx),  
           '3_mid': slice(0, nx),
           '3_top': slice(0, nx),
           '1_rec': slice(0, nx),
           }
                
            ysq = {}
            xsq = {}

            for kk in slc_y.keys():
                ysq[kk] = ylat[slc_y[kk], :]  
                xsq[kk] = xlon[:,slc_x[kk]] 

            figsize = (3, 3) #square
        else:
            figsize = (3, 8)  
        fig.set_size_inches(figsize)

        to_plot = fid.variables[ll][:]
        if ll == 'ssh_ins':
            vl = max(abs(vm), abs(vM))
        if ll == 'ssh_cos':
            vl = vl*0.1*0.5 #multiplying the cos fields with a factor of 20. 
        FigPth2 = os.path.join(FigPth, ll)
        if not os.path.exists(FigPth2):
            os.makedirs(FigPth2)

        for ii, tt in enumerate(time_array):  # snapshots
            picname_pre = '{0}-{1:05d}'.format(wpid, int(tt/3600))
            if squaresYN:
                for kk in slc_y.keys():
                    picpath = os.path.join(
                        FigPth2, '{0}-{1}.npz'.format(picname_pre, kk))  #hw changed .png to .npz
                    xaxis=xsq[kk]
                    yaxis=ysq[kk]
                    imagevar=to_plot[ii, slc_y[kk], slc_x[kk]]
                    #Normalize to [-1,1] (normalizing in this code seems to be the most convenient way.)
                    imagevar=imagevar/vl
                    imagevar=np.flipud(imagevar) #hw:Due to indexing in y, needs to flip vertically
                    
                    # Add "channels" dimension; otherwise some tensorflow functions would not work
                    imagevar = imagevar[..., np.newaxis]
                    
                    np.savez(picpath,imagevar=imagevar,vl=vl) #saving the normalization factor vl too.
   
            else:            
                xaxis=xlon
                yaxis=ylat
                imagevar= to_plot[ii, :, :]
                #Normalize to [-1,1] 
                imagevar=imagevar/vl
                imagevar=np.flipud(imagevar) #hw:Due to indexing in y, needs to flip vertically
                # Add "channels" dimension; otherwise some tensorflow functions would not work
                imagevar = imagevar[..., np.newaxis]
                if saveYN:
                    picpath = os.path.join(FigPth2, picname_pre+'.npz') #hw changed .png to .npz
                    #Note: didn't use .npy because it somehow can not save masked arrays. 
                    np.savez(picpath,imagevar=imagevar,vl=vl) 
                else:
                    pass  

        del to_plot  

    return


if __name__ == "__main__":
    saveYN = True  # set to True of you want to save figures
    squaresYN = True  # True: takes three square panels, down-jet, mid-jet and up-jet. Note that HW hasn't tested what will happen if this is False. 
    list_to_plt = ['ssh_ins', 'ssh_cos','ssh_lof']  # some variables dict

    # below are tuning parameters for the figures
    npix = 512  # number of pixels in x direction  (approximate) ( this variable is actually useless after HW's modifications)
 
    placeholder=argv[1]
    # %% Opening file --------------------------------------------------------|
    if len(argv) == 3:
        # if script not in folder, we specifiy where the experiments are
        # index 0 in the python script name
        Pth = os.path.abspath(os.path.realpath(argv[2]))
    else:
        # default is to assume script is in the folder containing experiments
        Pth =  os.path.abspath(os.path.realpath(
            os.path.join(__file__, os.pardir)))
    dsets = create_datasets_list(Pth)
    # above: calling both realpath and abspath solves problems with symlinks

    FigPth = os.path.join(Pth, placeholder)  # figs dir
    if os.path.isdir(FigPth):
        shutil.rmtree(FigPth)
    os.mkdir(FigPth)

    vm, vM = find_global_minmax(dsets, list_to_plt) #Find the max and min from ALL the .nc files available under Pth.


    for ds in dsets:
        process_one_dataset(ds, list_to_plt, vm, vM, FigPth, npix, placeholder)
