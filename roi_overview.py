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


from vgatPAG.database.db_tables import Trackings, AudioStimuli, VisualStimuli, Session, Roi, Recording, Mouse
from vgatPAG.paths import output_fld
from vgatPAG.variables import miniscope_fps, tc_colors, shelter_width_px


np.warnings.filterwarnings('ignore')

# ----------------------------------- Setup ---------------------------------- #
# ? Define some params
n_sec_pre = 2 
n_sec_post = 6

DEBUG = False

mice = Mouse.fetch("mouse")

# Get all sessions
sessions = {m:(Session & f"mouse='{m}'").fetch("sess_name") for m in mice}

# Get the recordings for each session
recordings = {m:{s:(Recording & f"sess_name='{s}'" & f"mouse='{m}'").fetch(as_dict=True) for s in sessions[m]} for m in mice}


def create_figure():
    ncols = 5
    nrows = 5
        
    f, axarr = plt.subplots(figsize=(25, 12), ncols=ncols, nrows=nrows)

    titles = ["Aligned to stim onset", "Aligned to escape onset", "Aligned to at shelter", "Aligned to spont homing", "Aligned to spont runs"]
    ylables = ['Shelter dist, px', 'Speed, px/frame', 'Ang vel, def/frame', 'Raw ROI sig', 'Mean ROI sig',]
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

def get_figure_path(mouse, sess, roi_id):
    f1 = os.path.join(output_fld, f"{mouse}")
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

def plot_roi_mean_sig(ax, traces, label=None):
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

    ax.plot(mean, lw=3, color=salmon, label=label)
    ax.fill_between(np.arange(frames_pre + frames_post), 
                                        mean-err, 
                                        mean+err, color=salmon, alpha=.3)

def get_spont_homings(shelter_distance, speed, roi_sig, astims, vstims):
    stims = astims + vstims

    in_roi = np.zeros_like(shelter_distance)
    in_roi[shelter_distance > 450] = 1
    in_shelt = np.zeros_like(shelter_distance)
    in_shelt[shelter_distance <=100] = 1
    speed_th = np.zeros_like(speed)
    speed_th[speed > 4] = 1

    ins, outs = get_times_signal_high_and_low(in_roi, th=.5, min_time_between_highs=30)
    ins_shelt, outs_shelt = get_times_signal_high_and_low(in_shelt, th=.5, min_time_between_highs=30)

    # f, ax = plt.subplots()
    # ax.plot(shelter_distance)

    starts, good_outs = [], []
    for n, roi_out in enumerate(outs):
        next_in_shelt = [i for i in ins_shelt if i>roi_out]
        if not next_in_shelt: continue
        else:
            next_in_shelt = next_in_shelt[0]

        if roi_out != outs[-1]:
            if next_in_shelt > outs[n+1]: continue # skip if there will be another out before an in shelter

        if next_in_shelt - roi_out > 2 * behav_fps: 
            continue # we only want fast events
        stims_range = [s for s in stims if np.abs(s - roi_out) < 6 * behav_fps] # we don't want stuff with stimuli involved
        if stims_range: continue

        fast = np.where(speed_th[:roi_out] == 1)[0][::-1]
        start = fast[0]
        for f in fast:
            if start - f <= 1:
                start = f
            else:
                break

        # Check if thers roi sig
        if np.std(roi_sig[start-200:next_in_shelt+200]) < 0.1: continue

        starts.append(start)
        good_outs.append(roi_out)

    # for start in starts:
    #     ax.axvline(start, color='r')
    # for start in good_outs:
    #     ax.axvline(start, color='g', alpha=.5)


    # plt.show()

    
    return starts

def get_roi_runs(shelter_distance, speed, roi_sig, astims, vstims):
    stims = astims + vstims

    in_roi = np.zeros_like(shelter_distance)
    in_roi[shelter_distance > 400] = 1
    in_shelt = np.zeros_like(shelter_distance)
    in_shelt[shelter_distance <=100] = 1
    speed_th = np.zeros_like(speed)
    speed_th[speed > 4] = 1

    ins, outs = get_times_signal_high_and_low(in_roi, th=.5, min_time_between_highs=frames_pre+frames_post+1)
    ins_shelt, outs_shelt = get_times_signal_high_and_low(in_shelt, th=.5, min_time_between_highs=frames_pre+frames_post+1)

    starts = []
    for shelt_out in outs_shelt:
        next_in_roi = [i for i in ins if i>shelt_out]
        if not next_in_roi: continue
        else:
            next_in_roi = next_in_roi[0]

        if next_in_roi - shelt_out > 6 * behav_fps: 
            continue # we only want fast events
        stims_range = [s for s in stims if np.abs(s - shelt_out) < 6 * behav_fps] # we don't want stuff with stimuli involved
        if stims_range: continue

        fast = np.where(speed_th[:shelt_out] == 1)[0][::-1]
        start = fast[0]
        for f in fast:
            if start - f <= 1:
                start = f
            else:
                break

        # Check if thers roi sig
        if np.std(roi_sig[start-200:next_in_roi+200]) < 0.1: continue

        starts.append(start)
    return starts



# ---------------------------------------------------------------------------- #
#                                  MAKE FIGUES                                 #
# ---------------------------------------------------------------------------- #

# Loop over sessions
for mouse in mice:
    for sess in tqdm(sessions[mouse]):
        # Create a figure for each ROI in the session

        nrois = []
        for rec in recordings[mouse][sess]:
            roi_ids, roi_sigs = (Roi & f"sess_name='{sess}'" & f"rec_name='{rec['rec_name']}'").fetch("roi_id", "signal", as_dict=False)
            nrois.append(len(roi_ids))
        if np.std(nrois) > 0:
            raise ValueError

        
        figures, axes, paths, n_trials = [], [], [], []
        stim_traces, escape_traces, shelt_traces = [], [], []
        dist_traces = dict(stim_aligned=[], escape_aligned=[], shelter_aligned=[])
        speed_traces = dict(stim_aligned=[], escape_aligned=[], shelter_aligned=[])
        spont_traces, spont_runs_traces = [], []
        for n, roi_id in enumerate(roi_ids):
            f, axs = create_figure()
            f_path = get_figure_path(mouse, sess, roi_id)
            figures.append(f)
            axes.append(axs)
            paths.append(f_path)
            n_trials.append(0)

            stim_traces.append([])
            escape_traces.append([])
            shelt_traces.append([])
            spont_traces.append([])
            spont_runs_traces.append([])

            for k,v in dist_traces.items(): v.append([])
            for k,v in speed_traces.items(): v.append([])

            if DEBUG: break

        # Loop over recordings
        for rec in  recordings[mouse][sess]:
            print(f"\n\n{mouse} - {sess} - {rec['rec_name']}")
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


            # Get ROI traces
            roi_ids, roi_sigs = (Roi & f"sess_name='{sess}'" & f"rec_name='{rec['rec_name']}'").fetch("roi_id", "signal", as_dict=False)
            roi_sigs = [list(r) for r in roi_sigs]
            nrois = len(roi_ids)
            
            # Get spont homings and runs
            spont_homings = get_spont_homings(shelter_distance, speed, roi_sigs[0], astims, vstims)
            spont_runs = get_roi_runs(shelter_distance, speed, roi_sigs[0], astims, vstims)

            # LOOP OVER ROIS
            for r, (roi_id, rsig) in enumerate(zip(roi_ids, roi_sigs)):
                f, (stim_aligned_axs, escape_aligned_axs, shelt_aligned_axs, spont_homing_axs, spont_runs_axs) = figures[r], axes[r]
                f_path = paths[r]
            
                for stims, colors in zip((astims, vstims), (astim_colors, vstim_colors)):
                    for stim, color in zip(stims, colors):
                        # Get times of events
                        try:
                            estart = np.where(speed[stim+5:stim+frames_post]>=6)[0][0]+stim+5
                        except:
                            continue
                        
                        try:
                            at_shelter = np.where(shelter_distance[stim:stim+frames_post] <= 0)[0][0] + stim
                        except:
                            continue

                        n_trials[r] += 1

                        # Plot aligned to stim onset
                        plot_traces(stim_aligned_axs, shelter_distance, speed, ang_vel, rsig, stim, color)
                        stim_traces[r].append(rsig[stim-frames_pre:stim+frames_post])
                        dist_traces['stim_aligned'][r].append(shelter_distance[stim-frames_pre : stim+frames_post])
                        speed_traces['stim_aligned'][r].append(speed[stim-frames_pre : stim+frames_post])

                        # Plot traces aligned to escape start
                        plot_traces(escape_aligned_axs, shelter_distance, speed, ang_vel, rsig, estart, color)
                        escape_traces[r].append(rsig[estart-frames_pre:estart+frames_post])
                        dist_traces['escape_aligned'][r].append(shelter_distance[stim-frames_pre:stim+frames_post])
                        speed_traces['escape_aligned'][r].append(speed[stim-frames_pre:stim+frames_post])

                        # Plot alined to at shelter
                        plot_traces(shelt_aligned_axs, shelter_distance, speed, ang_vel, rsig, at_shelter, color)
                        shelt_traces[r].append(rsig[at_shelter-frames_pre:at_shelter+frames_post])
                        dist_traces['shelter_aligned'][r].append(shelter_distance[stim-frames_pre:stim+frames_post])
                        speed_traces['shelter_aligned'][r].append(speed[stim-frames_pre:stim+frames_post])

                # Plot spont homings
                for spont in spont_homings:
                    plot_traces(spont_homing_axs, shelter_distance, speed, ang_vel, rsig, spont, magenta)
                    spont_traces[r].append(rsig[spont-frames_pre:spont+frames_post])

                # Plot spont runs
                for spont in spont_runs:
                    plot_traces(spont_runs_axs, shelter_distance, speed, ang_vel, rsig, spont, orange)
                    spont_runs_traces[r].append(rsig[spont-frames_pre:spont+frames_post])

                if DEBUG: break

        # Now plot averages
        for r, roi_id in enumerate(roi_ids):
            f, (stim_aligned_axs, escape_aligned_axs, shelt_aligned_axs, spont_homing_axs, spont_runs_axs) = figures[r], axes[r]

            # Plt mean for spont homings
            if spont_traces[r]:
                plot_roi_mean_sig(spont_homing_axs[4], spont_traces[r], label=f"{len(spont_traces[r])} spont. homings")
                spont_homing_axs[4].legend()

            # Plt mean for spont runs
            if spont_runs_traces[r]:
                plot_roi_mean_sig(spont_runs_axs[4], spont_runs_traces[r], label=f"{len(spont_runs_traces[r])} spont. runs")
                spont_runs_axs[4].legend()


            # Plot mean ROI traces
            if n_trials[r]:
                plot_roi_mean_sig(stim_aligned_axs[4], stim_traces[r])
                plot_roi_mean_sig(escape_aligned_axs[4], escape_traces[r])
                plot_roi_mean_sig(shelt_aligned_axs[4], shelt_traces[r])

            f.suptitle(f"{rec['mouse']} - {sess} - {roi_id} - {n_trials[r]} trials")

            if DEBUG: break
            else:
                save_figure(f, paths[r], verbose=True)                
                plt.close(f)
            if DEBUG: break
        if DEBUG: break

if DEBUG: plt.show()
# TODO fix spont detection