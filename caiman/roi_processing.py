# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rich.progress import track
import pickle
from pyinspect import search
import pyinspect as pi
from fcutils.plotting.utils import save_figure

from utils import load_tiff_video_caiman, get_component_trace_from_video

pi.install_traceback()
print('ready')

# %%

# ---------------------------------------------------------------------------- #
#                                   GET FILES                                  #
# ---------------------------------------------------------------------------- #

# fld = Path('/Users/federicoclaudi/Dropbox (UCL - SWC)/Project_vgatPAG/analysis/doric/BF161p1_dPAG/19JUN03')

fld = Path(r'D:\Dropbox (UCL - SWC)\Project_vgatPAG\analysis\doric\BF136p1_dPAG\19FEB04')

vid_name = '19FEB04_GCaMP_BF136p1_ds126_raw_crop2_mc_rig.tif'
bgsub_vid_name = '19FEB04_GCaMP_BF136p1_ds126_raw_crop2_mc_rig_bgsub.tif'

files = dict(
            raw=fld/vid_name, 
            bgsub=fld/bgsub_vid_name,
            A=fld/'A.npy',
            conts=fld/'all_contour_data.pkl'
        )

# check files exists
for n,f in files.items():
    if not f.exists():
        if n == 'bgsub':
            raise FileExistsError('No bg subtracted video, go make it in fiji! [rolling ball filter width: 10px]')
        else:
            raise FileExistsError(f'Could not find file {n}: {f.name}')

video = load_tiff_video_caiman(str(files['bgsub']))


# %%

# ---------------------------------------------------------------------------- #
#                               Load caiman model                              #
# ---------------------------------------------------------------------------- #
# Spatial components: in a d1 x d2 x n_components matrix
A = np.load(files['A'])

n_components = A.shape[-1]
good_components = [0,1, 3, 4, 13, 10, 9, 11, 15, 19, 17, 20, 21, 27, 29, 30, 38, 39, 28, 30, 31, 26, 25, 35, 32, 33, 31, 22, 14, 11, 24, 23, 40, 41]
print('good components: ', sorted(good_components))

isgood = [True if n in good_components else False
                            for n in np.arange(n_components)]


# Masks (a d1 x d1 x n_comp array with 1 only where each cell is )
mask_th = .04
masks = np.zeros_like(A)
masks[A > mask_th] = 1

with open(files['conts'], 'rb') as f:
    conts = pickle.load(f)


f, axarr = plt.subplots(ncols=3, figsize=(16, 9))
axarr[0].imshow(np.sum(A, 2))
axarr[1].imshow(1 - np.sum(masks, 2), cmap='gray_r')

# Get new masks
contours = dict(roi_idx=[], points=[])
new_masks = np.zeros((masks.shape[0], masks.shape[1], len(conts)))
for n, cont in enumerate(conts):
    if n not in good_components: 
        color = 'r'
        alpha=.7
        txtcolor=[.8, .2, .2]
    else:
        color = 'g'
        alpha=1
        txtcolor= [.4, .8, .4]

    # store good components
    if n in good_components:
        new_masks[:, :, n] = masks[:, :, n]
        contours['points'].append(cont['coordinates'])
        contours['roi_idx'].append(cont['neuron_id'])

    points = cont['coordinates']
    axarr[0].plot(*points.T, color=color, lw=2)
    axarr[1].plot(*points.T, color=color, lw=2, alpha=alpha)

    axarr[1].text(cont['CoM'][1], cont['CoM'][0], str(n), color=txtcolor, zorder=100)


axarr[2].imshow(1 - np.sum(new_masks, 2), cmap="gray_r")

axarr[0].set(title='A', xticks=[], yticks=[])
axarr[1].set(title=f'Masks - threhsold {mask_th}', xticks=[], yticks=[])

save_figure(f, str(fld / 'contours'))

# Save new masks and contours to file
np.save(str(fld / 'masks.npy'), new_masks)
pd.DataFrame(contours).to_hdf(str(fld / 'contours.h5'), key='hdf')

# %%

# ---------------------------------------------------------------------------- #
#                                EXTRACT TRACES                                #
# ---------------------------------------------------------------------------- #
n_frames  = video.shape[0]
traces = np.zeros((n_frames, n_components))

for compn in track(range(n_components)):
    traces[:, compn] = get_component_trace_from_video(compn, masks, n_frames, video)



# %%
# Save good traces
save_fld = fld / 'fiji_traces'
save_fld.mkdir(exist_ok=True)

for compn in track(range(n_components)):
    if not isgood[compn]: continue

    # save the mask
    f, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(1 - masks[:, :, compn], cmap='gray_r')
    ax.set(title=f'ROI {compn}', xticks=[], yticks=[])

    save_figure(f, str(save_fld/f'roi_{compn}_mask'), verbose=False)
    del f

    # save the trace
    with open(str(save_fld/f'ROI{compn}.txt'), 'w') as fl:
        for n in traces[:, compn]:
            fl.write(str(n)+'\n')

np.save(str(save_fld/'masks.npy'), masks)
np.save(str(save_fld/'traces.npy'), traces)

# %%
pi.ok('Data saved', save_fld.parent.name + '/' +  save_fld.name)

