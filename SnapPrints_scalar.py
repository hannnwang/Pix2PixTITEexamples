 #Editted from SnapPrints.py, but prints the scalar values directly as numpy arrays.
#Compared to Nico's panels, shifted the panels a little upwards to avoid wave forcing
from sys import argv  # handling of arguments
import os  # path manipulation
import shutil  # remove non-empty directory
import numpy as np
import matplotlib #HW: add this for non-interactive backend
matplotlib.use('Agg') #HW: add this for non-interactive backend
import matplotlib.pyplot as plt  # plotting functionalities
from joblib import Parallel, delayed
import multiprocessing
from netCDF4 import Dataset  # handling netcdf files


def create_datasets_list(abs_path_of_top):
    """ From the top directory (may be the local one actually), this function
    returns a list of all eligible absolute paths containing the appropriate
    files
    The operator would give the directory containing the jet_XXX directories
    IN: abs_path_of_top (string). Empty (default) compute abspath of parent
    OUT: list of absolute paths (['ap0', 'ap1', ...])
    """

    list_of_ncs = []
    for root, _, files in os.walk(abs_path_of_top):# HW: os.walk returns a tuple with the current_folder, a list of sub_folders,
    # and a list of files in the current_folder
        for ff in files:
            # print(ff)
            if ff.startswith('file_ave_'):  # for Aurelien's data
                list_of_ncs.append(os.path.join(root, ff)) #HW： adding a new item
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

"""
def plot_one(X, Y, Z, picpath, ax, fig, vl, cmap, dpi):
    plt.cla()  # clears axes, but preserves figure (incl. size)
    ax.pcolormesh(X, Y, Z, vmin=-vl, vmax=vl, ncolors=ncolorhw, cmap=cmap)#changed by hw
    ax.set_aspect('equal')
    ax.axis('off')
    if saveYN:
        fig.savefig(picpath, dpi=dpi)
    else:
        plt.draw()
        plt.pause(0.01)

    return
"""

def process_one_dataset(ds, list_to_plt, vm, vM, FigPth, npix, cmap):
    """ From the absolute path of an .nc file, print all of the snapshots"""
    fid = Dataset(ds, mode='r')

    print("Processing "+str(ds))

    iwp = ds.find('wp')  # index of the wp sub-string. What follows is wp
    wpid = ds[iwp:-3]
    """
    wpid = ds[iwp+2]
    try:  # hw: look for the second digit following wp and add 
        wpid += str(int(ds[iwp+3]))  # that way I check if the next item is int
    except ValueError:
        wpid += '0'  # pad with a zero
    """
    # %% Let's plot a couple of things ---------------------------------------|
    fig, ax = plt.subplots()

    for ll in list_to_plt:
        # different variables have slightly different coordinates; MAY CRAP OUT
        # IN THE FUTURE IF DIFFERENT FIELDS HAVE DIFFERENT SIZES
        coords = fid.variables[ll].coordinates.split()
        # the above yields a list, each element is a coordinate: t, x, y resp.
        # the below works because each coordinate above is also a variable
        time_array = fid.variables[coords[0]][:]
        xlon = fid.variables[coords[1]][:]
        ylat = fid.variables[coords[2]][:]
        ny, nx = xlon.shape
    
        if squaresYN:
            slc_y = {'3_bot': slice(104, 104+nx),  
           '3_mid': slice(104+nx//2, 104+nx//2+nx),
           '3_top': slice(104+nx, 104+2*nx),
           '1_rec': slice(104, 104+2*nx),
           #'9_a': slice(104, 104+nx),  
           #'9_b': slice(104+nx//8, 104+nx//8+nx),
           #'9_c': slice(104+nx//8*2, 104+nx//8*2+nx),
           #'9_d': slice(104+nx//8*3, 104+nx//8*3+nx),
           #'9_e': slice(104+nx//8*4, 104+nx//8*4+nx),
           #'9_f': slice(104+nx//8*5, 104+nx//8*5+nx),
           #'9_g': slice(104+nx//8*6, 104+nx//8*6+nx),
           #'9_h': slice(104+nx//8*7, 104+nx//8*7+nx),
           #'9_i': slice(104+nx//8*8, 104+nx//8*8+nx)
           
           } 
    
            slc_x={'3_bot': slice(0, nx),  
           '3_mid': slice(0, nx),
           '3_top': slice(0, nx),
           '1_rec': slice(0, nx),
           #'9_a': slice(0, nx),  
           #'9_b': slice(0, nx),
           #'9_c': slice(0, nx),
           #'9_d': slice(0, nx),
           #'9_e': slice(0, nx),
           #'9_f': slice(0, nx),
           #'9_g': slice(0, nx),
           #'9_h': slice(0, nx),
           #'9_i': slice(0, nx)
           }
                
            ysq = {}
            xsq = {}

            for kk in slc_y.keys():
                ysq[kk] = ylat[slc_y[kk], :]  # MA(ylat, msk[kk])
                xsq[kk] = xlon[:,slc_x[kk]] 

            figsize = (3, 3) #HW: square
        else:
            figsize = (3, 8)  # 4*xlon.shape[0]/xlon.shape[1])
        fig.set_size_inches(figsize)
        #dpi = int(npix/figsize[0]) 

        to_plot = fid.variables[ll][:]
        if ll == 'ssh_ins' or ll == 'ssh_lof':
            vl = max(abs(vm), abs(vM))
        if ll == 'ssh_cos' or ll == 'ssh_sin':
            # waves have much smaller amplitude so we divide. I believe it
            # is important that we multiply by a simple, independent number for
            # easy reinterpretation (e.g. ssh in m, waves in dm)
            # And while we can tune it, it is also important that the colormap
            # never saturates, therefore we can't divide by too much. 
            vl = max(abs(vm), abs(vM))*0.1*0.5 #******hw added a factor of 0.5 here because it's still gonna be bounded by 1. Check if it's consistent with other codes.
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
                    # Add "channels" dimension; otherwise some tf functions would not work
                    imagevar = imagevar[..., np.newaxis]
                    np.savez(picpath,imagevar=imagevar,vl=vl) #saving the normalization factor too.
                    #np.savez(picpath,imagevar=imagevar,xaxis=xaxis,yaxis=yaxis)  #**haven't compressed to 256*256 yet
                    #plot_one(xsq, ysq[kk], to_plot[ii, slc[kk], :],
                    #         picpath, ax, fig, vl, cmap, dpi)
            else:
                """
                plt.cla()  # clears axes, but preserves figure (including size)
                ax.pcolormesh(xlon, ylat, to_plot[ii, :, :],
                              vmin=-vl, vmax=vl, ncolors=ncolorhw, cmap=cmap) #***changed by hw
                ax.set_aspect('equal')
                ax.axis('off')
                """
                xaxis=xlon
                yaxis=ylat
                imagevar= to_plot[ii, :, :]
                #Normalize to [-1,1] (normalizing in this code seems to be the most convenient way.)
                imagevar=imagevar/vl
                imagevar=np.flipud(imagevar) #hw:Due to indexing in y, needs to flip vertically
                # Add "channels" dimension; otherwise some tf functions would not work
                imagevar = imagevar[..., np.newaxis]
                if saveYN:
                    picpath = os.path.join(FigPth2, picname_pre+'.npz') #hw changed .png to .npz
                    #Note: didn't use .npy because it somehow can not save masked arrays. 
                    #np.savez(picpath,imagevar=imagevar,xaxis=xaxis,yaxis=yaxis) 
                    np.savez(picpath,imagevar=imagevar,vl=vl) 
                    #fig.savefig(picpath, dpi=int(npix/figsize[0]))
                else:
                    pass
                    #plt.draw()
                    #plt.pause(0.01)

        del to_plot  

    return


if __name__ == "__main__":
    # %% This section is for adjustable parameters ---------------------------|
    saveYN = True  # set to True of you want to save figures
    squaresYN = True  # True: takes three snapshots, top, middle, bottom
    # if you don't save, the snapshots will be displayed in real time
    list_to_plt = ['ssh_ins', 'ssh_cos','ssh_lof', 'ssh_sin']  # some variables dict

    # below are tuning parameters for the figures
    npix = 512  # number of pixels in x direction  (approximate) (actually useless after hw's modifications)
    #cmap = argv[1]  # you can try different ones, see matplotlib website  #HW: first command line input; argvp[0] is the script name or something. 
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
    # print os.path.abspath(os.path.join(yourpath, os.pardir))
    dsets = create_datasets_list(Pth)
    # above: calling both realpath and abspath solves problems with symlinks

    FigPth = os.path.join(Pth, placeholder)  # figs dir
    if os.path.isdir(FigPth):
        shutil.rmtree(FigPth)
    os.mkdir(FigPth)

    vm, vM = find_global_minmax(dsets, list_to_plt)
  
    # num_processes = min([multiprocessing.cpu_count(), len(dsets)])
    # print("We have "+str(num_processes)+" cpus at our disposal")
    # Parallel(n_jobs=num_processes)(
    #     delayed(process_one_dataset)(ds, list_to_plt, vm, vM, FigPth, npix,
    #                                  cmap)
    #     for ds in dsets)

    for ds in dsets:
        process_one_dataset(ds, list_to_plt, vm, vM, FigPth, npix, placeholder)
