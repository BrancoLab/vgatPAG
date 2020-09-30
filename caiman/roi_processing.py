# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from fcutils.plotting.utils import save_figure

from movie_visualizer import compare_videos
from utils import start_server, load_tiff_video_caiman, get_component_trace_from_video

from caiman.source_extraction.cnmf.cnmf import load_CNMF


# %%

# ---------------------------------------------------------------------------- #
#                                   GET FILES                                  #
# ---------------------------------------------------------------------------- #

fld = Path('/Users/federicoclaudi/Dropbox (UCL - SWC)/Project_vgatPAG/analysis/doric/BF161p1_dPAG/19JUN03')

vid_name = '19JUN03_BF161p1_v1_ds126_crop_raw_mc_rig.tif'
bgsub_vid_name = '19JUN03_BF161p1_v1_ds126_crop_raw_mc_rig_bgsub.tif'
cnm_name = '19JUN03_BF161p1_v1_ds126_crop_ffcSub_cnm.hdf5'

files = dict(
            raw=fld/vid_name, 
            bgsub=fld/bgsub_vid_name,
            cnm=fld/cnm_name,
        )

# check files exists
for n,f in files.items():
    if not f.exists():
        raise FileExistsError(f'Could not find file {n}: {f.name}')


# inspect videos side by side
# compare_videos(raw=str(files['raw']), bgsub=str(files['bgsub']), notebook=False, contrast_limits=None)

# Get bg video data and params
n_frames = cnm.estimates.C.shape[1]
video = load_tiff_video_caiman(str(files['bgsub']))


# %%

# ---------------------------------------------------------------------------- #
#                               Load caiman model                              #
# ---------------------------------------------------------------------------- #

c, dview, n_processes = start_server()
cnm = load_CNMF(files['cnm'], n_processes=1, dview=dview)


# Get spatial components from model
n_components = cnm.estimates.A.shape[1] #  both good and bad components
good_components = cnm.estimates.idx_components
bad_components = cnm.estimates.idx_components_bad

isgood = [True if n in good_components else False
                            for n in np.arange(n_components)]

# Spatial components: in a d1 x d1 x n_components matrix
A = np.reshape(cnm.estimates.A.toarray(), list(cnm.estimates.dims)+[-1], order='F') # set of spatial footprints
centroids = cnm.estimates.center

# Masks (a d1 x d1 x n_comp array with 1 only where each cell is )
masks = np.zeros_like(A)
masks[A > .1] = 1

# TODO use: https://github.com/flatironinstitute/CaImAn/blob/87c46b76c06f32bc3a992ebee8c3ec5b90214094/caiman/utils/visualization.py#L324
# TODO make overlay video

# %%

# ---------------------------------------------------------------------------- #
#                                EXTRACT TRACES                                #
# ---------------------------------------------------------------------------- #

traces = np.zeros((n_frames, n_components))

for compn in tqdm(range(n_components)):
    traces[:, compn] = get_component_trace_from_video(compn, masks, n_frames, video)


# %%

# ---------------------------------------------------------------------------- #
#                                 PLOT RESULTS                                 #
# ---------------------------------------------------------------------------- #

f, axarr = plt.subplots(ncols=2, nrows=n_components, figsize=(8, 2*n_components),
                gridspec_kw={'width_ratios': [1, 3]})

for compn in tqdm(range(n_components)):
    color = [.4, .8, .4] if isgood[compn] else [.8, .4, .4]


    axarr[compn, 0].imshow(1 - masks[:, :, compn], cmap='gray_r')
    axarr[compn, 1].plot(traces[:, compn], lw=2, color=color)

    axarr[compn, 0].set(xticks=[], yticks=[])

    title = f'vidfile {vid_name} \ncnmf file {cnm_name}' if compn == 0 else f'ROI {compn}'
    axarr[compn, 1].set(title=title)

# %%
# Save good traces
save_fld = fld / 'fiji_traces'
save_fld.mkdir(exist_ok=True)

count = 0
for compn in tqdm(range(n_components)):
    if not isgood[compn]: continue

    # save the mask
    f, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(1 - masks[:, :, compn], cmap='gray_r')
    ax.set(title=f'ROI {count}', xticks=[], yticks=[])

    save_figure(f, str(save_fld/f'roi_{count}_mask'))
    del f

    # save the trace
    with open(str(save_fld/f'roi_{count}_mask.txt'), 'w') as fl:
        for n in traces[:, compn]:
            fl.write(str(n)+'\n')

    count += 1

np.save(str(save_fld/'masks.npy'), masks)
np.save(str(save_fld/'traces.npy'), traces)