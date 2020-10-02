# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from pyinspect import search

from fcutils.plotting.utils import save_figure

from movie_visualizer import compare_videos
from utils import start_server, load_tiff_video_caiman, get_component_trace_from_video

from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.utils.visualization import get_contours


# %%

# ---------------------------------------------------------------------------- #
#                                   GET FILES                                  #
# ---------------------------------------------------------------------------- #

# fld = Path('/Users/federicoclaudi/Dropbox (UCL - SWC)/Project_vgatPAG/analysis/doric/BF161p1_dPAG/19JUN03')

fld = Path(r'D:\Dropbox (UCL - SWC)\Project_vgatPAG\analysis\doric\BF161p1_dPAG\19JUN03')

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

# # Get bg video data and params
# c, dview, n_processes = start_server()
# cnm = load_CNMF(files['cnm'], n_processes=4, dview=dview)

video = load_tiff_video_caiman(str(files['bgsub']))


# %%

# ---------------------------------------------------------------------------- #
#                               Load caiman model                              #
# ---------------------------------------------------------------------------- #
# Get spatial components from model
n_components = cnm.estimates.A.shape[1] #  both good and bad components
# good_components = cnm.estimates.idx_components
# bad_components = cnm.estimates.idx_components_bad

good_components = [8, 0, 13, 37, 5, 12,  13,18, 19, 1, 22, 21, 35, 33, 24, 32, 26]

isgood = [True if n in good_components else False
                            for n in np.arange(n_components)]

# Spatial components: in a d1 x d2 x n_components matrix
A = np.reshape(cnm.estimates.A.toarray(), list(cnm.estimates.dims)+[-1], order='F') # set of spatial footprints
centroids = cnm.estimates.center

# Masks (a d1 x d1 x n_comp array with 1 only where each cell is )
mask_th = .04
masks = np.zeros_like(A)
masks[A > mask_th] = 1

# %%
conts = get_contours(cnm.estimates.A.toarray(), cnm.estimates.dims)

# conts = [c for c in conts if c['neuron_id'] in good_components]


f, axarr = plt.subplots(ncols=3, figsize=(16, 9))
axarr[0].imshow(np.sum(A, 2))
axarr[1].imshow(1 - np.sum(masks, 2), cmap='gray_r')

# Get new masks
contours = dict(roi_idx=[], points=[])
new_masks = np.zeros((masks.shape[0], masks.shape[1], len(conts)))
for n, cont in enumerate(conts):
    if n not in good_components: 
        color = 'r'
    else:
        color = 'g'

    # store good components
    if n in good_components:
        new_masks[:, :, n] = masks[:, :, n]
        contours['points'].append(cont['coordinates'])
        contours['roi_idx'].append(cont['neuron_id'])

    points = cont['coordinates']
    axarr[0].plot(*points.T, color=color, lw=2)
    axarr[1].plot(*points.T, color=color, lw=2)

    axarr[1].text(cont['CoM'][1], cont['CoM'][0], str(n), color='y')


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
    with open(str(save_fld/f'ROI{compn}.txt'), 'w') as fl:
        for n in traces[:, compn]:
            fl.write(str(n)+'\n')

    count += 1

np.save(str(save_fld/'masks.npy'), masks)
np.save(str(save_fld/'traces.npy'), traces)

# %%

# ---------------------------------------------------------------------------- #
#                                  Make video                                  #
# ---------------------------------------------------------------------------- #

# TODO make video with overlay
import cv2

masks = np.load(str(fld / 'masks.npy'))
conts = pd.read_hdf(str(fld / 'contours.h5'), key='hdf')
contours = []
for i, con in conts.iterrows():
    contours.append(np.floor(con.points)[2:-2, :].astype(np.int32))


overlay = np.zeros((masks.shape[0], masks.shape[1], 3))
overlay[:, :, 0] = masks.sum(2)


cont_overlay = np.zeros((masks.shape[0], masks.shape[1], 3))
cv2.drawContours(cont_overlay, contours, -1, (0, 255, 0), 1)


frames_fld = fld / 'frames'
frames_fld.mkdir(exist_ok=True)



for frame in tqdm(range(video.shape[0])):
    original = video[frame, :, :]  / 10

    img = np.zeros((original.shape[0], original.shape[1], 3))
    img[:, :, 0] = original / 1.5
    img[:, :, 1] = original / 1.5
    img[:, :, 2] = original / 1.5

    img = cv2.addWeighted(img,0.5,overlay,0.1,0)
    
    img = cv2.addWeighted(img,0.5,cont_overlay,0.005,0)


    img = cv2.resize(img, (img.shape[0] * 4, img.shape[1] * 4))



    # cv2.imshow('name', img)
    # cv2.waitKey(10)

    if frame < 10:
        n = f'0000{frame}'
    elif frame < 100:
        n = f'000{frame}'
    elif frame < 1000:
        n = f'00{frame}'
    elif frame < 10000:
        n = f'0{frame}'
    else:
        n = str(frame)

    cv2.imwrite(str(frames_fld / f'{n}.png'), img * 255)

#     videowriter.write(np.ones_like(added_image).astype(np.int32) * np.random.uniform(0, 200))
# videowriter.release()

# ffmpeg -i frames\%05d.png -c:v libx264 -vf fps=100 -pix_fmt yuv420p overlay2.mp4

# frames_fld.unlink()
# %%
