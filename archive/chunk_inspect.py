# %%
import numpy as np
import matplotlib.pyplot as plt
from rich.progress import track

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
from Analysis.misc import get_tiff_starts_ends, get_doric_chunk_baseline, get_chunk_rolling_mean_subracted_signal, get_chunked_dff
from vgatPAG.database.db_tables import RoiDFF, TiffTimes, Recording

from fcutils.plotting.utils import save_figure, clean_axes
from fcutils.maths.utils import derivative
from pathlib import Path
import shutil
from vedo.colors import colorMap

fld = Path('D:\\Dropbox (UCL)\\Project_vgatPAG\\analysis\\doric\\Fede\\ddf_tag_aligned')

DO = dict(
    plot_chunks_baseline=False,
    plot_chunks_smoothed=True,
)

# %%
"""
    Plot raw ad dff traces aligned to a tag
    for all recordings and all rois
"""


tag_type = 'VideoTag_B'
event_type =  ['Loom_Escape', 'US_Escape', 'LoomUS_Escape'] #


fps = 30

n_sec_pre = 4 # rel tag_type tag onset
n_sec_post = 4 # # rel tag_type tag onset
n_frames_pre = n_sec_pre * fps
n_frames_post = n_sec_post * fps

hanning = 6  #  width of hanning window used for smoothing 


# %%
"""
    Plot chunk baselines over the raw data
"""
for mouse, sess, sessname in track(mouse_sessions, description='plotting whole trace baseline view'):
    if not DO['plot_chunks_baseline']: break
    
    # Get data
    tracking, ang_vel, speed, shelter_distance, dffs, signals, nrois, is_rec, roi_ids = \
                        get_mouse_session_data(mouse, sess, sessions, hanning_window=hanning)
    tags = get_tags_by(mouse=mouse, sess_name=sess, event_type=event_type, tag_type=tag_type)

    # make figure
    f, axarr = plt.subplots(nrows=nrois, ncols=1,  figsize=(18, 4 * nrois), sharex=True)

    # loop over rois
    for n, (sig, rid) in enumerate(zip(signals, roi_ids)):
        # plot trace
        axarr[n].plot(sig, color='skyblue', lw=1, alpha=.6, label='raw signal')
        axarr[n].axhline(np.percentile(sig[is_rec==1], RoiDFF.DFF_PERCENTILE), color='k', 
                                    lw=1, alpha=.4, label=f'trace {RoiDFF.DFF_PERCENTILE}th [erc')

        # Plot speed trace
        axarr[n].plot(speed / (np.max(speed)/np.max(sig)), color='#f2c038', lw=.5, alpha=.35, zorder=-1, label='norm.speed')

        # Get chunks start end times
        tiff_starts, tiff_ends = get_tiff_starts_ends(is_rec)

        # loop over chunks
        for c, (start, end) in enumerate(zip(tiff_starts, tiff_ends)):
            th = np.percentile(sig[start:end], RoiDFF.DFF_PERCENTILE)
            high_th = np.percentile(sig[start:end], 50)
            low_th = np.percentile(sig[start:end], 10)

            axarr[n].plot([start, end],  [th, th], lw=2, color='m', label=f'{RoiDFF.DFF_PERCENTILE}th percentile' if c == 0 else None)
            axarr[n].plot([start, end],  [high_th, high_th], lw=1, ls='--', color='m', label='10th and 50th percentiles' if c == 0 else None)
            axarr[n].plot([start, end],  [low_th, low_th], lw=1, ls='--', color='m')

            # mark stimuli
            tgs = tags.loc[(tags.stim_frame < end)&(tags.stim_frame > start)]
            for stim_frame in tgs.stim_frame.values:
                if not is_rec[stim_frame]:
                    continue
                axarr[n].plot([stim_frame, stim_frame], [th, sig[stim_frame]], color='k', lw=2, alpha=.8)
                axarr[n].scatter(stim_frame, sig[stim_frame], color='k', s=100, zorder=99)

        axarr[n].legend()
        axarr[n].set(title=rid, ylabel='raw signal')

        # break

    # cleanup and save
    axarr[-1].set(xlabel='frames')
    
    clean_axes(f)
    f.tight_layout()
    save_figure(f, fld / (sessname + '_doric_chunks_baselines'))
    break

# %%
"""
    Plot each ROI's signal with the chunk smoothed by a rolling mean filter
"""
window = 600
for mouse, sess, sessname in track(mouse_sessions, description='plotting smoothed chunks'):
    if not DO['plot_chunks_smoothed']: break
    
    # Get data
    tracking, ang_vel, speed, shelter_distance, dffs, signals, nrois, is_rec, roi_ids = \
                        get_mouse_session_data(mouse, sess, sessions, hanning_window=hanning)
    tags = get_tags_by(mouse=mouse, sess_name=sess, event_type=event_type, tag_type=tag_type)

    # make figure
    f, axarr = plt.subplots(nrows=nrois, ncols=2,  figsize=(18, 4 * nrois), sharex=True)

    # Get chunks start end times
    tiff_starts, tiff_ends = get_tiff_starts_ends(is_rec)

    # loop over rois
    for n, (sig, rid) in enumerate(zip(signals, roi_ids)):

        # get chunked dff
        dff = get_chunked_dff(sig, tiff_starts, tiff_ends)

        # Get smooth signal
        diff, smooth = get_chunk_rolling_mean_subracted_signal(dff, tiff_starts, tiff_ends, window=window)

        # Plot        
        axarr[n, 0].plot(dff, color='k', label='dff')
        axarr[n, 0].plot(smooth, lw=3, color='salmon', label='smoothed dff')

        axarr[n, 1].plot(diff, lw=1, color='seagreen')

        #  TODO compute stds only for is_rec==1
        axarr[n, 1].axhspan(-np.std(diff), np.std(diff), 
                                color='seagreen', 
                                alpha=.3, hatch='/', zorder=110, label='subtracted dff std')
        axarr[n, 1].axhspan(-np.std(dff), np.std(dff), 
                                color='k', 
                                alpha=.1,zorder=110, label='dff std')
        axarr[n, 0].legend()
        axarr[n, 1].legend()
        # break


    # cleanup and save
    axarr[0, 0].set(title=f'{sessname} - Filter window width: {window} frames')
    axarr[0, 1].set(title='"raw" dff - smoothed dff')
    axarr[-1, 0].set(xlabel='frames')
    
    clean_axes(f)
    f.tight_layout()
    save_figure(f, fld / (sessname + f'_doric_chunks_smooted_window_{window}'))
    # break



# %%
f, ax = plt.subplots(figsize=(12, 8))
ax.plot(diff, LW=2)
# ax.plot(dff, lw=1, ls='--')
# ax.plot(smooth)
# ax.set(xlim=[124180, 124200])
# ax.set(xlim=[94620, 94800])
# ax.set(xlim=[46300, 46500])
# ax.set(xlim=[10000, 20000])

# %%
# f, ax = plt.subplots(figsize=(12, 8))
# ax.plot(sig, lw=3, ls='--')
# ax.plot(derivative(is_rec))
# ax.set(xlim=[94620, 94800])

# # %%

# %%
