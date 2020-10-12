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

fld = Path(r'D:\Dropbox (UCL - SWC)\Project_vgatPAG\analysis\doric\BF166p3_dPAG\19JUN19')

vid_name = '19JUN19_BF166p3_ds126_crop3_raw_mc_rig.tif'
bgsub_vid_name = '19JUN19_BF166p3_ds126_crop3_raw_mc_rig_bgsub.tif'

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
good_components = [7, 35, 8, 26, 25, 1, 0, 19, 21, 29, 28, 23, 20, 22, 37, 30, 31, 32, 36, 27, 33]
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

# ---------------------------------------------------------------------------- #
#                                 PLOT RESULTS                                 #
# ---------------------------------------------------------------------------- #

f, axarr = plt.subplots(ncols=2, nrows=n_components, figsize=(8, 2*n_components),
                gridspec_kw={'width_ratios': [1, 3]})

for compn in track(range(n_components)):
    color = [.4, .8, .4] if isgood[compn] else [.8, .4, .4]


    axarr[compn, 0].imshow(1 - masks[:, :, compn], cmap='gray_r')
    axarr[compn, 1].plot(traces[:, compn], lw=2, color=color)

    axarr[compn, 0].set(xticks=[], yticks=[])

    title = f'vidfile {vid_name}' if compn == 0 else f'ROI {compn}'
    axarr[compn, 1].set(title=title)

# %%
# Save good traces
save_fld = fld / 'fiji_traces'
save_fld.mkdir(exist_ok=True)

count = 0
for compn in track(range(n_components)):
    if not isgood[compn]: continue

    # save the mask
    f, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(1 - masks[:, :, compn], cmap='gray_r')
    ax.set(title=f'ROI {count}', xticks=[], yticks=[])

    save_figure(f, str(save_fld/f'roi_{count}_mask'), verbose=False)
    del f

    # save the trace
    with open(str(save_fld/f'ROI{compn}.txt'), 'w') as fl:
        for n in traces[:, compn]:
            fl.write(str(n)+'\n')

    count += 1

np.save(str(save_fld/'masks.npy'), masks)
np.save(str(save_fld/'traces.npy'), traces)

pi.ok('Data saved', save_fld.parent.name + '/' +  save_fld.name)

# %%

# ---------------------------------------------------------------------------- #
#                                  Make video                                  #
# ---------------------------------------------------------------------------- #

# # TODO make video with overlay
# import cv2

# masks = np.load(str(fld / 'masks.npy'))
# conts = pd.read_hdf(str(fld / 'contours.h5'), key='hdf')
# contours = []
# for i, con in conts.iterrows():
#     contours.append(np.floor(con.points)[2:-2, :].astype(np.int32))


# overlay = np.zeros((masks.shape[0], masks.shape[1], 3))
# overlay[:, :, 0] = masks.sum(2)


# cont_overlay = np.zeros((masks.shape[0], masks.shape[1], 3))
# cv2.drawContours(cont_overlay, contours, -1, (0, 255, 0), 1)


# frames_fld = fld / 'frames'
# frames_fld.mkdir(exist_ok=True)



# for frame in track(range(video.shape[0])):
#     original = video[frame, :, :]  / 10

#     img = np.zeros((original.shape[0], original.shape[1], 3))
#     img[:, :, 0] = original / 1.5
#     img[:, :, 1] = original / 1.5
#     img[:, :, 2] = original / 1.5

#     img = cv2.addWeighted(img,0.5,overlay,0.1,0)
    
#     img = cv2.addWeighted(img,0.5,cont_overlay,0.005,0)


#     img = cv2.resize(img, (img.shape[0] * 4, img.shape[1] * 4))



#     # cv2.imshow('name', img)
#     # cv2.waitKey(10)

#     if frame < 10:
#         n = f'0000{frame}'
#     elif frame < 100:
#         n = f'000{frame}'
#     elif frame < 1000:
#         n = f'00{frame}'
#     elif frame < 10000:
#         n = f'0{frame}'
#     else:
#         n = str(frame)

#     cv2.imwrite(str(frames_fld / f'{n}.png'), img * 255)

#     videowriter.write(np.ones_like(added_image).astype(np.int32) * np.random.uniform(0, 200))
# videowriter.release()

# ffmpeg -i frames\%05d.png -c:v libx264 -vf fps=100 -pix_fmt yuv420p overlay2.mp4

# frames_fld.unlink()
# %%
