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
from fcutils.maths.utils import derivative
from scipy.stats import zscore

from myterial import light_blue_light, purple_light


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
    if len(tags) < 4:
        continue

    # Get doric chunks times
    tiff_starts, tiff_ends = get_tiff_starts_ends(is_rec)

    # Create figure
    f, axarr = plt.subplots(nrows=nrois, ncols=1,  figsize=(18, 4 * nrois), sharex=True)


    # Loop over rois
    for n, (dff, sig, rid) in enumerate(zip(dffs, signals, roi_ids)):

        # ------------------------------- Get roi data ------------------------------- #

        # loop over tagged events
        roi_sigs, total_sig = [], []
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
            DORIC_CHUNK_DFF = zscore((RAW - doric_chunk_th)/doric_chunk_th)
            
            total_sig.append((sig-doric_chunk_th)/doric_chunk_th)
            roi_sigs.append(DORIC_CHUNK_DFF)

        # ------------------------------- Get threshold ------------------------------ #

        # stack data
        roi_sigs = np.vstack(roi_sigs)


        # Get recording threshold
        s = np.concatenate(total_sig)
        th = np.mean(s) + 1 * np.std(s)
        thl = np.mean(s) - 1 * np.std(s)
        axarr[n].axhspan(thl, th, color=light_blue_light, alpha=.3, hatch='/', zorder=110, label='threshold')

        # ----------------------------------- Plot ----------------------------------- #

        # Plot mean signal
        axarr[n].plot(roi_sigs.T, color=[.9, .9, .9], zorder=-1)

        mean = np.mean(roi_sigs, 0)
        std = np.std(roi_sigs, 0)
        plot_mean_and_error(mean, std, axarr[n], color='salmon')
        axarr[n].plot(mean, color=[.2, .2, .2], zorder=-1, lw=5)

        axarr[n].axvline(n_frames_pre, lw=2, color=[.3, .3, .3])
        axarr[n].set(ylabel=f'{rid}\nchunk DFF signal')
        axarr[n].legend()
        # break

    axarr[0].set(title=sessname)
    axarr[-1].set(xlabel='frames')

    clean_axes(f)
    f.tight_layout()
    save_figure(f, fld / (sessname))

    break


