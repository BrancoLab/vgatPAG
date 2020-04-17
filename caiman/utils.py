
from fcutils.file_io.utils import check_file_exists, get_file_name, check_create_folder, listdir
from fcutils.file_io.io import load_yaml

import matplotlib.pyplot as plot
import os
import numpy as np
import cv2
import logging

import caiman as cm
from caiman.source_extraction.cnmf.cnmf import load_CNMF

# ---------------------------------- Loaders --------------------------------- #
def load_fit_cnfm(model_path, n_processes, dview):
    return load_CNMF(model_path, n_processes=n_processes, dview=dview)   

def get_mc_files_from_fld(fld):
    # Get motion corrected files
    rigid_mc = [f for f in listdir(fld) if "_rig__" in f and f.endswith("_.mmap") 
                        and "_order_C" in f and "d2" in f and "d1" in f]

    pw_mc = [f for f in listdir(fld) if "_els__" in f and f.endswith("_.mmap") 
                        and "_order_C" in f and "d2" in f and "d1" in f]
    
    if len(rigid_mc) > 1 or len(pw_mc) > 1:
        raise ValueError("Found too many files!")
    elif not rigid_mc or not pw_mc:
        raise ValueError("didn't find enough files")
    else:
        rigid_mc = rigid_mc[0]
        pw_mc = pw_mc[0]

    return rigid_mc, pw_mc


def load_fit_cnfm_and_data(fld, n_processes, dview, mc_type="els"):
    """
        Loads a saved CNMF-E model and necessary data from a mouse's folder
    """
    check_create_folder(fld, raise_error=True)

    # Load CNMF-E model
    model_filepath = os.path.join(fld, "cnmfe_fit.hdf5")
    check_file_exists(model_filepath, raise_error=True)

    cnm = load_fit_cnfm(model_filepath, n_processes, dview)
    cnm.estimates.dims = cnm.dims

    # Get motion corrected files
    rigid_mc, pw_mc = get_mc_files_from_fld(fld)

    if mc_type == "els":
        video = pw_mc
    elif mc_type == "rig":
        video = rigid_mc
    else:
        raise ValueError("Unrecognized mc type argument")

    # Load memmapped motion corrected video
    Yr, dims, T = cm.load_memmap(video)
    images = Yr.T.reshape((T,) + dims, order='F')

    # Create a smoothed bg frame 
    kernel = np.ones((5,5),np.float32)
    bg = np.median(images, axis=0)
    smooth_bg = cv2.filter2D(bg, -1, kernel)

    # Get filtered and peak over noise ratio images
    cn_filter, pnr = cm.summary_images.correlation_pnr(images, gSig=cnm.params.init['gSig'][0], swap_dim=False) 

    return cnm, model_filepath, Yr, dims, T, images, smooth_bg, cn_filter, pnr


def load_tiff_video_caiman(videofile):
    check_file_exists(videofile, raise_error=True)
    return cm.load(videofile)

def load_mmap_video_caiman(videofile):
    check_file_exists(videofile, raise_error=True)

    Yr, dims, T = cm.load_memmap(videofile)
    images = Yr.T.reshape((T,) + dims, order='F')
    return Yr, dims, T, images


# ------------------------------- Params utils ------------------------------- #
def load_params(fld, **kwargs):
    params_file = os.path.join(fld, "01_PARAMS", "params.yml")
    return load_yaml(params_file)

def add_to_params_dict(params, **kwargs):
    for k,v in kwargs.items():
        params[k] = v
    return params

# -------------------------------- Small utils ------------------------------- #
def print_cnmfe_components(cnm, msg=None):
    print(f'\n\n ***** {msg}')
    print('Number of total components: ', len(cnm.estimates.C))

    try:
        print('Number of accepted components: ', len(cnm.estimates.idx_components))
    except :
        print("No accepted components yet\n")

def log_cnmfe_components(cnm, msg=None):
    try:
        clean = f'Number of accepted components: {len(cnm.estimates.idx_components)}'
    except :
        clean = "No accepted components yet\n"

    string = f"COMPONENTS - {msg}\n" + f"'Number of total components: {len(cnm.estimates.C)}\n" + clean
    logging.info(string)




def start_server(n_processes=24):
    # number of process to use, if you go out of memory try to reduce this one

    try:
        cm.stop_server()  # stop it if it was running
    except():
        pass

    try:
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                        n_processes=n_processes,  
                                                        single_thread=False)
    except:
        print("A cluster is already running")
        return None, None, None
    return c, dview, n_processes



# --------------------------------- Plotters --------------------------------- #
def plot_components_over_image(img, ax, coords, lw, good_components, cmap="viridis", only=None):
    """
        Plot the ROIs contours over an iamge to visualise the spatial location of the ROIs

        :param img: 2d or 3d numpy array with image data
        :param ax: matplotlib axis onto which stuff should be plotted
        :param coords: output of running cm.utils.visualization.get_contours
        :param lw: int, line width of ROIs
        :param good_components: list of components indices. The "good components are drawn in green and bad ones in red
    """

    # Display image
    ax.imshow(img, cmap=cmap)

    # Plot contours and component id
    for c in coords:
        if c['neuron_id'] in good_components:
            color = 'g'
        else:
            color = 'r'

        if only is not None:
            if c['neuron_id'] in good_components and only != "good":
                continue
            elif c['neuron_id'] not in good_components and only != "bad":
                continue

        ax.plot(*c['coordinates'].T, c=color, lw=lw)
        ax.scatter(c['CoM'][1], c['CoM'][0], s=400, color='k', zorder=80)
        ax.text(c['CoM'][1]-1, c['CoM'][0]+1, str(c['neuron_id']), color='w', zorder=99)