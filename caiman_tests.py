# %%
# imports
import matplotlib.pyplot as plt
import numpy as np
import os

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
import cv2

try:
    cv2.setNumThreads(0)
except:
    pass

from fcutils.video.utils import open_cvwriter, get_cap_from_images_folder, save_videocap_to_video
from fcutils.plotting.utils import fig2data 
from tqdm import tqdm


# %%

# ------------------------------- Set up files ------------------------------- #

#Paths to raw and ffc. 
fld = "D:\\Dropbox (UCL - SWC)\\Project_vgatPAG\\analysis\\doric\\BF164p1\\19JUN05"
fnames    = [os.path.join(fld, '19JUN05_BF164p1_v1_ds126_crop_ffcSub.tif')]  # ffc filename to be processed
fname_raw = [os.path.join(fld, '19JUN05_BF164p1_v1_ds126_crop_raw.tif')]  ## raw file

memmapped = os.path.join(fld, "memmap__d1_109_d2_92_d3_1_order_C_frames_22662_.mmap")

# start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
if 'dview' in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

# %%

# ---------------------------------------------------------------------------- #
#                                SETUP PARAMTERS                               #
# ---------------------------------------------------------------------------- #

# dataset dependent parameters
frate = 10.                       # movie frame rate
decay_time = 2.              # length of a typical transient in seconds

#static parameters
#pw_rigid = True  # flag for performing rigid or piecewise rigid motion correction
border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)

# motion correction parameters
mc_dict = {
    'gSig_filt': (2,2), # 2 seems best, 3 is good for poor SNR/high BG
    'strides': (30,30), # Can make it shakey
    'overlaps': (11,11), #
    'max_shifts': (25,25), # max shifts in rigid. makes a difference. usually 5-8. can be larger if motion is very rigid
    'max_deviation_rigid': 7, # non-rigid shift deviation from rigid max shifts. makes a difference
    'fnames': fnames,
    'fr': frate,
    'decay_time': decay_time,
    'border_nan': border_nan,
    'shifts_opencv': True,
    'num_frames_split':100,
    'nonneg_movie': True
}

opts = params.CNMFParams(params_dict=mc_dict)


# parameters for source extraction and deconvolution
p = 1               # order of the autoregressive system
K = 15  ##changed          # upper bound on number of components per patch, in general None
gSig = (2.5, 2.5)     #2.5  # gaussian width of a 2D gaussian kernel, which approximates a neuron. 
gSiz = (8, 8)     # average diameter of a neuron, in general 4*gSig+1. 
Ain = None          # possibility to seed with predetermined binary masks
merge_thresh = .6   # default 0.7. merging threshold, max correlation allowed
rf = 30             # default 40. half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
stride_cnmf = 15     # amount of overlap between the patches in pixels (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
tsub = 2            # downsampling factor in time for initialization, increase if you have memory problems
ssub = 1            # downsampling factor in space for initialization, increase if you have memory problems you can pass them here as boolean vectors
low_rank_background = None  # None leaves background of each patch intact, True performs global low-rank approximation if gnb>0
gnb = 0             # number of background components (rank) if positive, else exact ring model with following settings
#                         gnb= 0: Return background as b and W
#                         gnb=-1: Return full rank background B
#                         gnb<-1: Don't return background
nb_patch = 0        # number of background components (rank) per patch if gnb>0, else it is set automatically
min_corr = .8       # min peak value from correlation image
min_pnr = 6        # min peak to noise ration from PNR image
ssub_B = 2          # additional downsampling factor in space for background
ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor
bord_px = 0


opts.change_params(params_dict={'method_init': 'corr_pnr',  # use this for 1 photon
                                'K': K,
                                'gSig': gSig,
                                'gSiz': gSiz,
                                'merge_thresh': merge_thresh,
                                'p': p,
                                'tsub': tsub,
                                'ssub': ssub,
                                'rf': rf,
                                'stride': stride_cnmf,
                                'only_init': True,    # set it to True to run CNMF-E
                                'nb': gnb,
                                'nb_patch': nb_patch,
                                'method_deconvolution': 'oasis',       # could use 'cvxpy' alternatively
                                'low_rank_background': low_rank_background,
                                'update_background_components': True,  # sometimes setting to False improve the results
                                'min_corr': min_corr,
                                'min_pnr': min_pnr,
                                'normalize_init': False,               # just leave as is
                                'center_psf': True,                    # leave as is for 1 photon
                                'ssub_B': ssub_B,
                                'ring_size_factor': ring_size_factor,
                                'del_duplicates': True,                # whether to remove duplicates from initialization    
                                'border_pix': bord_px})                # number of pixels to not consider in the borders)

# %%

# ---------------------------------------------------------------------------- #
#                                   ANALYSIS                                   #
# ---------------------------------------------------------------------------- #

# -------------------------- Load and filter MC data ------------------------- #

Yr, dims, T = cm.load_memmap(memmapped)
images = Yr.T.reshape((T,) + dims, order='F')
cn_filter, pnr = cm.summary_images.correlation_pnr(images, gSig=gSig[0], swap_dim=False)
# inspect_correlation_pnr(cn_filter, pnr)



# %%

# -------------------------------- FIT CNMF_E -------------------------------- #
cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=Ain, params=opts)
cnm.fit(images)






# %%

# ---------------------------------------------------------------------------- #
#               !                 QUALITY CONTROL                               #
# ---------------------------------------------------------------------------- #

# the components are evaluated in three ways:
#   a) the shape of each component must be correlated with the data
#   b) a minimum peak SNR is required over the length of a transient
#   c) each shape passes a CNN based classifier

min_SNR = 3             # adaptive way to set threshold on the transient size. 3 default.
r_values_min = 0.85     # threshold on space consistency (
                        # if you lower more components will be accepted, potentially with worst quality). 0.85 default. 0.6 seems ok.
#                        
cnm.params.set('quality', {'min_SNR': min_SNR,
                           'rval_thr': r_values_min,
                           'use_cnn': False})
cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

print(' ***** ')
print('Number of total components: ', len(cnm.estimates.C))
print('Number of accepted components: ', len(cnm.estimates.idx_components))


#%%
# ---------------------------- Inspect components ---------------------------- #

# plot contour plots of accepted and rejected components
import numpy as np
bg = np.median(images, axis=0)
cnm.estimates.plot_contours(img=cn_filter, idx=cnm.estimates.idx_components)
cnm.estimates.idx_components

# plt components over bakground image

def plot_components_over_image(img, ax, coords):
    # Plt image
    ax.imshow(img, cmap="gray")

    # Plot contours and component id
    for c in coords:
        if c['neuron_id'] in good_compontents:
            color = 'g'
        else:
            color = 'r'
        lw=15

        ax.plot(*c['coordinates'].T, c=color, lw=lw)
        ax.text(c['CoM'][1], c['CoM'][0], str(c['neuron_id']), color=color)



coordinates = cm.utils.visualization.get_contours(cnm.estimates.A, bg.shape, thr=.2, thr_method="max")
good_compontents = cnm.estimates.idx_components

f, axarr = plt.subplots(figsize=(15, 10), ncols=2)

for ax, im, ttl in zip(axarr, [bg, cn_filter], ["mean signal", "cn_filter"]):
    plot_components_over_image(im, ax, coordinates)



# %%
# ? Make video with contour over the images
tot_frames, w, h = images.shape
frames = np.arange(0, tot_frames, 100) # add only every N frames

fld = "D:\\Dropbox (UCL - SWC)\\Project_vgatPAG\\analysis\\doric\\Fede\\frames"
plt.ioff()

kernel = np.ones((5,5),np.float32)
smooth_bg = cv2.filter2D(bg,-1,kernel/25)

# Create frames
if True:
    for n, fnum in tqdm(enumerate(frames)):
        f, ax = plt.subplots(figsize=(120, 100), dpi=5)
        img = cv2.filter2D(images[fnum, :, :].copy(), -1, kernel/121)
        plot_components_over_image(img/smooth_bg, ax, coordinates)
        # save_figure(f, os.path.join(fld, f"{n}"), verbose=False)
        f.savefig(os.path.join(fld, f"{n}"), dpi=5)
        plt.close()
        

# Stitch the frames in a video
""" run this in ffmpeg from the correct folder to get the video out
ffmpeg -i frames\%1d.png -c:v libx264 -vf fps=10 -pix_fmt yuv420p out.mp4
"""
