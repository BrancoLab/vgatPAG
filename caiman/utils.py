
from fcutils.file_io.utils import check_file_exists, get_file_name, check_create_folder, listdir

import matplotlib.pyplot as plot
import os
import numpy as np
import cv2

import caiman as cm
from caiman.source_extraction.cnmf.cnmf import load_CNMF

# ---------------------------------- Loaders --------------------------------- #
def load_fit_cnfm(model_path, n_processes, dview):
    return load_CNMF(model_path, n_processes=n_processes, dview=dview)   

def load_fit_cnfm_and_data(fld, n_processes, dview, mc_type="els"):
    """
        Loads a saved CNMF-E model and necessary data from a mouse's folder
    """
    check_create_folder(fld, raise_error=True)

    # Load CNMF-E model
    model_filepath = os.path.join(fld, "cnmfe_fit.hdf5")
    check_file_exists(model_filepath, raise_error=True)

    cnm = load_fit_cnfm(model_filepath, n_processes, dview)

    # Get motion corrected files
    try:
        rigid_mc = [f for f in listdir(fld) if "_rig__" in f and f.endswith("_.mmap") 
                            and "_order_C" in f and "d2" in f and "d1" in f][0]

        pw_mc = [f for f in listdir(fld) if "_els__" in f and f.endswith("_.mmap") 
                            and "_order_C" in f and "d2" in f and "d1" in f][0]
    except:
        raise ValueError("Could not load rigid and piecewise motion corrected videos. Make sure to run motion correction")
    
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




# -------------------------------- Small utils ------------------------------- #
def print_cnmfe_components(cnm, msg=None):
    print(f'\n\n ***** {msg}')
    print('Number of total components: ', len(cnm.estimates.C))

    try:
        print('Number of accepted components: ', len(cnm.estimates.idx_components))
    except :
        print("No accepted components yet")



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
        ax.text(c['CoM'][1], c['CoM'][0], str(c['neuron_id']), color=color)