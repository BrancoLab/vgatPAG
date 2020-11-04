# %%
from tqdm import tqdm
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
from vgatPAG.database.db_tables import RoiDFF, TiffTimes

from fcutils.plotting.utils import save_figure, clean_axes
from pathlib import Path
import shutil
from vedo.colors import colorMap

fld = Path('D:\\Dropbox (UCL)\\Project_vgatPAG\\analysis\\doric\\Fede\\ddf_tag_aligned')

# %%
def get_tiff_starts_emds(mouse, sess):
    '''
        Gets the start and end time of doric chunks
        for a session
    '''
    tiff_starts, tiff_ends = (TiffTimes & f'mouse="{mouse}"' & f"sess_name='{sess}'").fetch("starts", "ends")
    last_ts = [0] + [t[-1] for t in tiff_starts][:-1]
    last_te = [0] + [t[-1] for t in tiff_ends][:-1]

    tiff_starts = np.concatenate([t+last_ts[n] for n,t in enumerate(tiff_starts)])
    tiff_ends = np.concatenate([t+last_te[n] for n,t in enumerate(tiff_ends)])

    return tiff_starts, tiff_ends

def get_doric_chunk_baseline(tiff_starts, tiff_ends, start, end, signal):
    # baseline raw for whole doric chunk
    tf = [tf for tf in tiff_starts if tf < start][-1]
    te = [te for te in tiff_ends if te > end][0]

    if te<tf: raise ValueError

    return np.percentile(signal[tf:te], RoiDFF.DFF_PERCENTILE)

def scatter_trial_baseline(ax, trace, trace_n, n_traces):
    c = colorMap(trace_n, name='bwr', vmin=0, vmax=n_traces)
    ax.scatter(0, np.mean(trace), color=c, edgecolors='k', lw=1, s=150, zorder=99)

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

# loop over each recording
for mouse, sess, sessname in track(mouse_sessions):
    # Get data
    tracking, ang_vel, speed, shelter_distance, dffs, signals, nrois, is_rec, roi_ids = \
                        get_mouse_session_data(mouse, sess, sessions, hanning_window=hanning)
    
    # Get tags metadata
    tags = get_tags_by(mouse=mouse, sess_name=sess, event_type=event_type, tag_type=tag_type)

    # Get doric chunks times
    tiff_starts, tiff_ends = get_tiff_starts_emds(mouse, sess)

    # Create figure
    f, axarr = plt.subplots(nrows=nrois, ncols=4,  figsize=(18, 4 * nrois), sharex=True)

    # Loop over rois
    for n, (dff, sig, rid) in enumerate(zip(dffs, signals, roi_ids)):

        # loop over tagged events
        for count, (i, tag) in enumerate(tags.iterrows()):
            start = tag.session_frame - n_frames_pre
            end = tag.session_frame + n_frames_post

            # Prepare traces for plotting
            RAW = sig[start:end]
            WHOLE_TRACE_DFF = dff[start:end]  # DFF with threshold computed on whole session

            doric_chunk_th = get_doric_chunk_baseline(tiff_starts, tiff_ends, start, end, sig)
            DORIC_CHUNK_DFF = (RAW - doric_chunk_th)/doric_chunk_th

            baseline_th =  np.percentile(sig[start:tag.session_frame], RoiDFF.DFF_PERCENTILE)
            BASELINE_DFF = (RAW - baseline_th) / baseline_th

            # scatter baseline mean signal
            scatter_trial_baseline(axarr[n, 0], RAW[:n_frames_pre], count, len(tags))
            scatter_trial_baseline(axarr[n, 1], WHOLE_TRACE_DFF[:n_frames_pre], count, len(tags))

            # Plot signals
            axarr[n, 0].plot(RAW, color='skyblue', lw=2, alpha=.6)
            axarr[n, 1].plot(WHOLE_TRACE_DFF, color='salmon', lw=2, alpha=.6)
            axarr[n, 2].plot(DORIC_CHUNK_DFF, color='seagreen', lw=2, alpha=.6)
            axarr[n, 3].plot(BASELINE_DFF, color='magenta', lw=2, alpha=.6)

 

        # Mark tag onset
        for ax in axarr[n, :]:
            ax.axvline(n_frames_pre, lw=2, color='r')

        # make titles, axis labels etc
        ts = [f' RAW Tag: {tag_type} ', ' WHOLE SESS DFF', ' DORIC CHUNK DFF', ' BASELINE DFF']
        ttls = [rid + t for t in ts]
        ylbs = ['raw', 'dff', 'dff', 'dff']

        for a, ax in enumerate(axarr[n, :]):
            ax.set(title=ttls[a], ylabel=ylbs[a])
    
    # Set X labels
    for ax in axarr[:, -1]:
        ax.set(xlabel='frames')
    
    # clean up and save
    clean_axes(f)
    f.tight_layout()
    save_figure(f, fld / (sessname))

    break

# %%
