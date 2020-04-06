import sys
sys.path.append("./")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.patches as patches
from scipy.stats import zscore, sem

from fcutils.plotting.colors import colorMap, desaturate_color
from fcutils.plotting.colors import *
from fcutils.plotting.utils import create_figure, clean_axes, save_figure, set_figure_subplots_aspect, forceAspect
from fcutils.plotting.plot_elements import plot_shaded_withline
from fcutils.objects import sort_list_of_strings
from fcutils.file_io.utils import check_create_folder
from fcutils.maths.utils import rolling_mean
from scipy.signal import medfilt as median_filter


from behaviour.utilities.signals import get_times_signal_high_and_low


from vgatPAG.database.db_tables import Trackings, AudioStimuli, VisualStimuli, Session, Roi, Recording
from vgatPAG.paths import output_fld
from vgatPAG.variables import miniscope_fps, tc_colors, shelter_width_px

# ----------------------------------- Setup ---------------------------------- #
# ? Define some params
n_sec_pre = 2 
n_sec_post = 6


# Get all sessions
sessions = Session.fetch("sess_name")

# Get the recordings for each session
recordings = {s:(Recording & f"sess_name='{s}'").fetch(as_dict=True) for s in sessions}


def create_figure():
    ncols = 3
    nrows = 6
        
    f, axarr = plt.subplots(figsize=(22, 12), ncols=ncols, nrows=nrows)

    titles = ["Aligned to stim onset", "Aligned to escape onset", "Aligned to at shelter"]
    ylables = ['Shelter dist, px', 'Speed, px/frame', 'Ang vel, def/frame', 'Raw ROI sig', 'Mean ROI sig', 'correlation']
    all_axes = []
    for i, title in enumerate(titles):
        axes = []
        for n, ylabel in enumerate(ylables):
            ax = axarr[n, i]
            if n == 0:
                ax.set(title=title)
            if i == 0:
                ax.set(ylabel=ylabel)
            if n == nrows-1:
                ax.set(xlabel="Time (s)")
            if ylabel=='correlation':
                ax.set(ylim=[-1, 1])
                ax.axhline(0, lw=1, ls=":", color='k', alpha=.5)
            axes.append(ax)
        all_axes.append(axes)
    
    set_figure_subplots_aspect(left=0.05, right=0.95, bottom=0.05, top=0.93, wspace=0.1, hspace=0.3)
    clean_axes(f)

    return f, all_axes

def adjust_ticks(ax, pre_frames, post_frames, fps, every=2):
    xticks = np.arange(0, pre_frames, every*fps)
    xticks_labels = [-int(pre_frames/fps - x/fps) for x in xticks]

    xticks2 = np.arange(pre_frames, post_frames+pre_frames, every*fps)
    xticks_labels2 = [int(x/fps - pre_frames/fps) for x in xticks2]
    ax.set(xticks=np.concatenate([xticks, xticks2]), xticklabels=np.concatenate([xticks_labels, xticks_labels2]))

def get_figure_path(rec, sess, roi_id):
    f1 = os.path.join(output_fld, f"{rec['mouse']}")
    check_create_folder(f1)
    fld = os.path.join(f1, sess)
    check_create_folder(fld)
    return os.path.join(fld, f"{roi_id}")

def plot_traces(axes, shelter_distance, speed, ang_vel, rsig, frame, color):
    axes[0].plot(np.arange(0, frames_pre), 
                            shelter_distance[frame-frames_pre:frame], color=color, lw=1, alpha=.35)
    axes[0].plot(np.arange(frames_pre, frames_pre+frames_post),
                            shelter_distance[frame:frame+frames_post], color=color, lw=1, alpha=.7)

    axes[1].plot(np.arange(0, frames_pre), 
                            speed[frame-frames_pre:frame], color=color, lw=1, alpha=.35)
    axes[1].plot(np.arange(frames_pre, frames_pre+frames_post),
                            speed[frame:frame+frames_post], color=color, lw=1, alpha=.7)

    axes[2].plot(np.arange(0, frames_pre), 
                            ang_vel[frame-frames_pre:frame], color=color, lw=1, alpha=.35)
    axes[2].plot(np.arange(frames_pre, frames_pre+frames_post),
                            ang_vel[frame:frame+frames_post], color=color, lw=1, alpha=.7)
    
    axes[3].plot(rsig[frame-frames_pre:frame+frames_post], color=color, lw=1.5, alpha=.3)

    for ax in axes:
        adjust_ticks(ax, frames_pre, frames_post, behav_fps, every=1)
        ax.axvline(frames_pre, lw=2, color='k', alpha=.7)

def plot_roi_mean_sig(ax, traces):
    mean, err = np.mean(traces, axis=0), sem(traces, axis=0)

    shape = np.array(traces).shape
    shuffled = np.array(traces).ravel()
    np.random.shuffle(shuffled)
    shuffled = shuffled.reshape(shape)

    shuff_mean, shuff_err = np.mean(shuffled, axis=0), sem(shuffled, axis=0)

    ax.plot(shuff_mean, lw=1, color=silver, alpha=.6)
    ax.fill_between(np.arange(frames_pre + frames_post), 
                                        shuff_mean-shuff_err,  
                                        shuff_mean+shuff_err, color=silver, alpha=.1)

    ax.plot(mean, lw=3, color=salmon)
    ax.fill_between(np.arange(frames_pre + frames_post), 
                                        mean-err, 
                                        mean+err, color=salmon, alpha=.3)

def plot_corr(x, y, ax, color, label=None, legend=False):
    x = np.nanmean(x, axis=0)
    y = np.nanmean(y, axis=0)

    df = pd.DataFrame(dict(x=x, y=y))
    r_window_size = 12
    # Interpolate missing data.
    df_interpolated = df.interpolate()
    # Compute rolling window synchrony
    rolling_r = df_interpolated['x'].rolling(window=r_window_size, center=True).corr(df_interpolated['y'])

    ax.plot(rolling_r, lw=1.5, color=color, label=label, alpha=.4)
    if legend: ax.legend()


# ---------------------------------------------------------------------------- #
#                                  MAKE FIGUES                                 #
# ---------------------------------------------------------------------------- #

# Loop over sessions
for sess in tqdm(sessions):
    # ----------------------------- Get sessiond data ---------------------------- #
    for rec in  recordings[sess]:
        behav_fps = rec['fps_behav']
        frames_pre = n_sec_pre * behav_fps
        frames_post = n_sec_post * behav_fps


        # Get stimuli
        vstims = (VisualStimuli & f"sess_name='{sess}'" & f"rec_name='{rec['rec_name']}'").fetch("frame")
        astims = (AudioStimuli & f"sess_name='{sess}'" & f"rec_name='{rec['rec_name']}'").fetch("frame")

        # Keep only stimuli that didn't happen immediately one after the other
        if np.any(vstims):
            vstims = [vstims[0]] + [s for n,s in enumerate(vstims[1:]) if s-vstims[n] > frames_post+1]
        if np.any(astims):
            astims = [astims[0]] + [s for n,s in enumerate(astims[1:]) if s-astims[n] > frames_post+1]
        astims = [s for s in astims if s not in vstims]

        # Get colors for each trial based on stim type
        vstim_colors = [seagreen for i in range(len(vstims))]
        astim_colors = [salmon for i in range(len(astims))]

        # Get tracking
        body_tracking = np.vstack((Trackings * Trackings.BodyPartTracking & f"sess_name='{sess}'"
                                    & f"rec_name='{rec['rec_name']}'" & "bp='body'").fetch1("x", "y", "speed"))
        ang_vel1 = median_filter((Trackings * Trackings.BodySegmentTracking & f"sess_name='{sess}'"
                                    & f"rec_name='{rec['rec_name']}'" & "bp1='neck'" & "bp2='body'").fetch1("angular_velocity"), kernel_size=11)

        ang_vel2 = median_filter((Trackings * Trackings.BodySegmentTracking & f"sess_name='{sess}'"
                                    & f"rec_name='{rec['rec_name']}'" & "bp1='body'" & "bp2='tail_base'").fetch1("angular_velocity"), kernel_size=11)

        ang_vel = np.median(np.vstack([ang_vel1, ang_vel2]), axis=0)
        speed = median_filter(body_tracking[2, :], kernel_size=11)
        shelter_distance = body_tracking[0, :]-shelter_width_px-200


        # Get ROIs
        roi_ids, roi_sigs = (Roi & f"sess_name='{sess}'" & f"rec_name='{rec['rec_name']}'").fetch("roi_id", "signal", as_dict=False)
        roi_sigs = [list(r) for r in roi_sigs]
        nrois = len(roi_ids)
        
        # LOOP OVER ROIS
        for r, (roi_id, rsig) in enumerate(zip(roi_ids, roi_sigs)):
            f, (stim_aligned_axs, escape_aligned_axs, shelt_aligned_axs) = create_figure()
            f_path = get_figure_path(rec, sess, roi_id)
           
            stim_traces, escape_traces, shelt_traces = [], [], []
            dist_traces = dict(stim_aligned=[], escape_aligned=[], shelter_aligned=[])
            speed_traces = dict(stim_aligned=[], escape_aligned=[], shelter_aligned=[])
            n_trials = 0
            for stims, colors in zip((astims, vstims), (astim_colors, vstim_colors)):
                for stim, color in zip(stims, colors):
                    # Get times of events
                    try:
                        estart = np.where(speed[stim+5:stim+frames_post]>=4)[0][0]+stim+5
                    except:
                        continue
                    
                    try:
                        at_shelter = np.where(shelter_distance[stim:stim+frames_post] <= 0)[0][0] + stim
                    except:
                        continue

                    n_trials += 1

                    # Plot aligned to stim onset
                    plot_traces(stim_aligned_axs, shelter_distance, speed, ang_vel, rsig, stim, color)
                    stim_traces.append(rsig[stim-frames_pre:stim+frames_post])
                    dist_traces['stim_aligned'].append(shelter_distance[stim-frames_pre : stim+frames_post])
                    speed_traces['stim_aligned'].append(speed[stim-frames_pre : stim+frames_post])

                    # Plot traces aligned to escape start
                    plot_traces(escape_aligned_axs, shelter_distance, speed, ang_vel, rsig, estart, color)
                    escape_traces.append(rsig[estart-frames_pre:estart+frames_post])
                    dist_traces['escape_aligned'].append(shelter_distance[stim-frames_pre:stim+frames_post])
                    speed_traces['escape_aligned'].append(speed[stim-frames_pre:stim+frames_post])

                    # Plot alined to at shelter
                    plot_traces(shelt_aligned_axs, shelter_distance, speed, ang_vel, rsig, at_shelter, color)
                    shelt_traces.append(rsig[at_shelter-frames_pre:at_shelter+frames_post])
                    dist_traces['shelter_aligned'].append(shelter_distance[stim-frames_pre:stim+frames_post])
                    speed_traces['shelter_aligned'].append(speed[stim-frames_pre:stim+frames_post])



            # Plot mean ROI traces
            if n_trials:
                plot_roi_mean_sig(stim_aligned_axs[4], stim_traces)
                plot_roi_mean_sig(escape_aligned_axs[4], escape_traces)
                plot_roi_mean_sig(shelt_aligned_axs[4], shelt_traces)


                # Plot correlations
                plot_corr(stim_traces, dist_traces['stim_aligned'], stim_aligned_axs[5], firebrick, label="Shelter distance")
                plot_corr(stim_traces, speed_traces['stim_aligned'], stim_aligned_axs[5], saddlebrown, label="Speed", legend=True)

                plot_corr(escape_traces, dist_traces['escape_aligned'], escape_aligned_axs[5], firebrick, label="Shelter distance")
                plot_corr(escape_traces, speed_traces['escape_aligned'], escape_aligned_axs[5], saddlebrown, label="Speed", legend=True)

                plot_corr(stim_traces, dist_traces['shelter_aligned'], shelt_aligned_axs[5], firebrick, label="Shelter distance")
                plot_corr(stim_traces, speed_traces['shelter_aligned'], shelt_aligned_axs[5], saddlebrown, label="Speed", legend=True)

            f.suptitle(f"{rec['mouse']} - {sess} - {roi_id} - {n_trials} trials")
            # save_figure(f, f_path, verbose=True)                
            # plt.close(f)
            break
        break
    break
plt.show()