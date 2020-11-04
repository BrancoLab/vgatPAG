# %%
import numpy as np
import matplotlib.pyplot as plt
from rich.progress import track
from pathlib import Path

from Analysis.misc import get_tiff_starts_ends, get_doric_chunk_baseline, percentile, get_doric_chunks
from vgatPAG.database.db_tables import RoiDFF, TiffTimes, Recording
from Analysis  import (
        mice,
        sessions,
        recordings,
        mouse_sessions,
        get_mouse_session_data,
)
from Analysis.tag_aligned import (
    manual_tags,
    get_tags_by,
    get_next_tag,
    get_last_tag,
)

from fcutils.plotting.plot_elements import plot_mean_and_error
from fcutils.plotting.utils import save_figure, clean_axes

# %%
# ---------------------------------- params ---------------------------------- #
fld = Path('D:\\Dropbox (UCL)\\Project_vgatPAG\\analysis\\doric\\Fede\\escape_selective')



tag_type = 'VideoTag_B'
event_type =  ['Loom_Escape', 'US_Escape', 'LoomUS_Escape'] #


fps = 30

n_sec_pre = 4 # rel tag_type tag onset
n_sec_post = 4 # # rel tag_type tag onset
n_frames_pre = n_sec_pre * fps
n_frames_post = n_sec_post * fps

hanning = 6  #  width of hanning window used for smoothing 

# %%
'''
    For each ROI and each tagged event, find frames that are > mean + 2*std
    of the signal during the doric chunk that the event belongs to.

    Also plot the mean signal across traces
'''
# loop over each recording
for mouse, sess, sessname in track(mouse_sessions, description='plotting aligned traces'):
    # Get data
    tracking, ang_vel, speed, shelter_distance, dffs, signals, nrois, is_rec, roi_ids = \
                        get_mouse_session_data(mouse, sess, sessions, hanning_window=hanning)
    
    # Get tags metadata
    tags = get_tags_by(mouse=mouse, sess_name=sess, event_type=event_type, tag_type=tag_type)

    # Get doric chunks times
    tiff_starts, tiff_ends = get_tiff_starts_ends(is_rec)

    # Create figure
    f, axarr = plt.subplots(nrows=nrois, ncols=1,  figsize=(18, 4 * nrois), sharex=True)
    # f, axarr = plt.subplots(nrows=2, ncols=1,  figsize=(18, 8), sharex=True)

    # Loop over rois
    for n, (dff, sig, rid) in enumerate(zip(dffs, signals, roi_ids)):

        # loop over tagged events
        roi_sigs = []
        for count, (i, tag) in enumerate(tags.iterrows()):
            if not is_rec[tag.session_frame]:
                continue
                # raise ValueError

            start = tag.session_frame - n_frames_pre
            end = tag.session_frame + n_frames_post

            x_pre = np.arange(n_frames_pre+1)
            x_post = n_frames_pre + x_pre[:-1]

            # Prepare traces for plotting
            RAW = sig[start:end]
            doric_chunk_th = get_doric_chunk_baseline(tiff_starts, tiff_ends, start, sig)
            DORIC_CHUNK_DFF = (RAW - doric_chunk_th)/doric_chunk_th
            roi_sigs.append(DORIC_CHUNK_DFF)

            # Get signal within the doric chunk
            chunk_start, chunk_end = get_doric_chunks(tiff_starts, tiff_ends, start)
            chunk = (sig[chunk_start:chunk_end] - doric_chunk_th)/doric_chunk_th

            # Get a threshold for the chunk
            high_th = np.mean(chunk) + 2 * np.std(chunk)
            low_th = np.mean(chunk) - 2 * np.std(chunk)
            above = np.where(DORIC_CHUNK_DFF > high_th)[0]
            below = np.where(DORIC_CHUNK_DFF < low_th)[0]

            # Plot stuff
            # axarr[n].axhline(high_th, lw=2, color='r', zorder=-1, alpha=.3)
            for selected in (above, below):
                axarr[n].scatter(selected, DORIC_CHUNK_DFF[selected], 
                        color='skyblue', s=25, zorder=10, lw=.5, edgecolors=[.2, .2, .2])

            axarr[n].plot(x_pre, DORIC_CHUNK_DFF[:n_frames_pre+1], color=[.8, .8, .8], zorder=-1)
            axarr[n].plot(x_post, DORIC_CHUNK_DFF[n_frames_pre:], color=[.7, .7, .7], zorder=-1)

        # Plot mean signal
        if not roi_sigs: 
            continue

        mean = np.mean(np.vstack(roi_sigs), 0)
        std = np.std(np.vstack(roi_sigs), 0)
        plot_mean_and_error(mean, std, axarr[n], color='salmon')
        axarr[n].plot(mean, color=[.2, .2, .2], zorder=-1, lw=5)

        axarr[n].axvline(n_frames_pre, lw=2, color=[.3, .3, .3])
        # break
    # break
# %%
f, ax = plt.subplots()
ax.plot(sig, color='k')
ax.axvline(chunk_start)
ax.axvline(chunk_end)

ax.axvline(tag.session_frame, color='red')
ax.axvline(tag.stim_frame, color='g')

ax.set(xlim=[chunk_start - 1000, chunk_end + 1000])
# %%
