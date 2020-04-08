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


from vgatPAG.database.db_tables import Trackings, AudioStimuli, VisualStimuli, Session, Roi, Recording, Mouse, TiffTimes
from vgatPAG.paths import output_fld
from vgatPAG.variables import miniscope_fps, tc_colors, shelter_width_px


np.warnings.filterwarnings('ignore')



# ----------------------------------- Setup ---------------------------------- #
# ? Define some params
n_sec_pre = 2 
n_sec_post = 8
n_sec_post_plot = 5
n_sec_spont_events = 2

DEBUG = True

mice = Mouse.fetch("mouse")

# Get all sessions
sessions = {m:(Session & f"mouse='{m}'").fetch("sess_name") for m in mice}

# Get the recordings for each session
recordings = {m:{s:(Recording & f"sess_name='{s}'" & f"mouse='{m}'").fetch(as_dict=True) for s in sessions[m]} for m in mice}





# ------------------------------ Plotting Utils ------------------------------ #
def create_figure(xmin, xmax):
    ncols = 7
    nrows = 5
        
    f, axarr = plt.subplots(figsize=(25, 12), ncols=ncols, nrows=nrows, sharey='row')

    titles = ["Aligned to stim onset", "Aligned to escape onset", "Aligned to speed peak", 
                "Aligned to at shelter", "Aligned to spont homing", "Aligned to spont homing max speed", "Aligned to spont runs",]
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
            if "Ang vel" in ylabel:
                ax.set(ylim=[-20, 20])
            
            ax.set(xlim=[xmin, xmax])
            axes.append(ax)
        all_axes.append(axes)
    
    set_figure_subplots_aspect(left=0.06, right=0.97, bottom=0.05, top=0.93, wspace=0.17, hspace=0.4)
    clean_axes(f)

    return f, all_axes

def adjust_ticks(ax, pre_frames, post_frames, fps, every=2):
    xticks = np.arange(0, pre_frames, every*fps)
    xticks_labels = [-int(pre_frames/fps - x/fps) for x in xticks]

    xticks2 = np.arange(pre_frames, post_frames+pre_frames, every*fps)
    xticks_labels2 = [int(x/fps - pre_frames/fps) for x in xticks2]
    ax.set(xticks=np.concatenate([xticks, xticks2]), xticklabels=np.concatenate([xticks_labels, xticks_labels2]))

    ax.axvline(frames_pre, lw=1.5, color=blackboard, alpha=.7)
    ax.axvspan(frames_pre-fps, frames_pre+fps, facecolor=desaturate_color(deepskyblue), alpha=0.075, zorder=-1)


def get_figure_path(mouse, sess, roi_id):
    f1 = os.path.join(output_fld, f"{mouse}")
    check_create_folder(f1)
    fld = os.path.join(f1, sess)
    check_create_folder(fld)
    return os.path.join(fld, f"{roi_id}")

def plot_traces(axes, shelter_distance, speed, ang_vel, rsig, frame, color):
    # Plot shelter distance
    axes[0].plot(np.arange(0, frames_pre), 
                            shelter_distance[frame-frames_pre:frame], color=color, lw=1, alpha=.35)
    axes[0].plot(np.arange(frames_pre, frames_pre+frames_post),
                            shelter_distance[frame:frame+frames_post], color=color, lw=1, alpha=.7)

    # Plot speed
    axes[1].plot(np.arange(0, frames_pre), 
                            speed[frame-frames_pre:frame], color=color, lw=1, alpha=.35)
    axes[1].plot(np.arange(frames_pre, frames_pre+frames_post),
                            speed[frame:frame+frames_post], color=color, lw=1, alpha=.7)

    # Plot angular velocity
    axes[2].plot(np.arange(0, frames_pre), 
                            ang_vel[frame-frames_pre:frame], color=color, lw=1, alpha=.35)
    axes[2].plot(np.arange(frames_pre, frames_pre+frames_post),
                            ang_vel[frame:frame+frames_post], color=color, lw=1, alpha=.7)
    
    # Plot ROI activity traces
    axes[3].plot(rsig[frame-frames_pre:frame+frames_post], color=color, lw=1.5, alpha=.3)

def plot_roi_mean_sig(ax, traces, random_traces, label=None):
    # Plot the mean trace and the mean of the randomly sampled trace
    mean, err = np.mean(traces, axis=0), sem(traces, axis=0)

    try:
        rmean, rerr = np.mean(random_traces, axis=0), sem(random_traces, axis=0)
    except:
        raise ValueError

    ax.plot(rmean, lw=1, color=silver, alpha=.6)
    ax.fill_between(np.arange(frames_pre + frames_post), 
                                        rmean-rerr,  
                                        rmean+rerr, color=silver, alpha=.2)

    ax.plot(mean, lw=3, color=salmon, label=label)
    ax.fill_between(np.arange(frames_pre + frames_post), 
                                        mean-err, 
                                        mean+err, color=salmon, alpha=.3)

    # Compute and plot the pre vs post event osnet stuff
    pre = np.nanmean(mean[frames_pre - behav_fps:frames_pre])
    post = np.nanmean(mean[frames_pre:frames_pre + behav_fps])

    pre_vs_post = (pre-post) / (pre+post)
    ax.set(title=r"$\frac{pre-post}{pre+post} = "+ str(round(pre_vs_post, 2)) +"$")
    # y = ax.get_ylim()[0]+2
    # if pre_vs_post < 0:
    #     rec = patches.Rectangle((y, frames_pre + (behav_fps*pre_vs_post)), width=np.abs(behav_fps*pre_vs_post), height=5,
    #                 facecolor=blackboard, alpha=.8)
    # else:
    #     rec = patches.Rectangle((y, frames_pre), width=np.abs(behav_fps*pre_vs_post), height=5,
    #                 facecolor=blackboard, alpha=.8)
    # ax.add_artist(rec)


def plot_roi_correlation(roi_sig, speed, ax):
    ax.scatter(roi_sig, speed, color=silver, s=5, alpha=.25)



# ---------------------------------- Getters --------------------------------- #

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

        if next_in_shelt - roi_out > n_sec_spont_events * behav_fps: 
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

        # Check that there's no errors
        if shelter_distance[start] < 450: continue

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

        if next_in_roi - shelt_out > (n_sec_spont_events+2) * behav_fps: 
            continue # we only want fast events

        # Get onset
        fast = np.where(speed_th[:shelt_out] == 1)[0][::-1]
        start = fast[0]
        for f in fast:
            if start - f <= 1:
                start = f
            else:
                break

        # Check if thers roi sig
        if np.std(roi_sig[start-frames_pre:start]) < 0.2: continue

        # Check that there's no errors
        if shelter_distance[start] > 100: continue

        starts.append(start)
    return starts

def get_roi_random_trace(rec, roi_sig):
    is_recording = (TiffTimes & f"rec_name='{rec['rec_name']}'").fetch1("is_ca_recording")
    rsig = np.array(roi_sig)[np.where(is_recording)]
    start = np.random.randint(1000, len(rsig)-1000)
    return rsig[start-frames_pre:start+frames_post]

def get_whole_roi_trace(roi_id, roi_sig, rec_name):
    is_recording = (TiffTimes & f"rec_name='{rec['rec_name']}'").fetch1("is_ca_recording")
    return np.array(roi_sig)[np.where(is_recording)], is_recording


# ---------------------------------------------------------------------------- #
#                                  MAKE FIGUES                                 #
# ---------------------------------------------------------------------------- #
# with plt.xkcd(): # ! use this for funzies
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
        stim_traces, escape_traces, speed_traces, shelt_traces = [], [], [], []
        random_traces = []
        spont_traces, spont_speed_traces, spont_runs_traces = [], [], []
        for n, roi_id in enumerate(roi_ids):
            behav_fps = recordings[mouse][sess][0]['fps_behav']
            xmax = n_sec_post_plot * behav_fps
            xmin = 0

            f, axs = create_figure(xmin, xmax)
            f_path = get_figure_path(mouse, sess, roi_id)
            figures.append(f)
            axes.append(axs)
            paths.append(f_path)
            n_trials.append(0)

            stim_traces.append([])
            escape_traces.append([])
            speed_traces.append([])
            shelt_traces.append([])
            spont_traces.append([])
            spont_speed_traces.append([])
            spont_runs_traces.append([])
            random_traces.append([])

            if DEBUG: break

        # Whole traces holders
        whole_traces = {r:[] for r in roi_ids}
        whole_traces['speed'] = []
        whole_traces['shelt_dist'] = []

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


            # Keep whole traces
            for roi_id, roi_sig in zip(roi_ids, roi_sigs):
                trace, is_recording = get_whole_roi_trace(roi_id, roi_sig, rec['rec_name'])
                whole_traces[roi_id].extend(list(trace))
            whole_traces['speed'].extend(list(speed[np.where(is_recording)]))
            whole_traces['shelt_dist'].extend(list(shelter_distance[np.where(is_recording)]))


            # LOOP OVER ROIS
            for r, (roi_id, rsig) in enumerate(zip(roi_ids, roi_sigs)):
                f, (stim_aligned_axs, escape_aligned_axs, speed_aligned_axs, shelt_aligned_axs, spont_homing_axs, spont_homing_speed_axs, spont_runs_axs,) = figures[r], axes[r]
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

                        if stim < frames_pre+1 or estart < frames_pre+1: continue # too soon too soon

                        speed_peak = np.argmax(np.nan_to_num(speed[stim:stim+frames_post])) + stim

                        n_trials[r] += 1

                        # Plot aligned to stim onset
                        plot_traces(stim_aligned_axs, shelter_distance, speed, ang_vel, rsig, stim, color)
                        stim_traces[r].append(rsig[stim-frames_pre:stim+frames_post])

                        # Plot traces aligned to escape start
                        plot_traces(escape_aligned_axs, shelter_distance, speed, ang_vel, rsig, estart, color)
                        escape_traces[r].append(rsig[estart-frames_pre:estart+frames_post])

                        # Plot aligned to speed peak
                        plot_traces(speed_aligned_axs, shelter_distance, speed, ang_vel, rsig, speed_peak, color)
                        speed_traces[r].append(rsig[speed_peak-frames_pre:speed_peak+frames_post])

                        # Plot alined to at shelter
                        plot_traces(shelt_aligned_axs, shelter_distance, speed, ang_vel, rsig, at_shelter, color)
                        shelt_traces[r].append(rsig[at_shelter-frames_pre:at_shelter+frames_post])

                        # Get ROI random sample
                        random_traces[r].append(list(get_roi_random_trace(rec, rsig)))

                # Plot spont homings
                for spont in spont_homings:
                    if spont < frames_pre + 1: continue
                    if shelter_distance[spont + frames_post + frames_pre] > 100: continue

                    # Plot aligned to spont homing onset
                    plot_traces(spont_homing_axs, shelter_distance, speed, ang_vel, rsig, spont, magenta)
                    spont_traces[r].append(rsig[spont-frames_pre:spont+frames_post])

                    # Plot aligned to spont homing max speed
                    speed_onset = np.argmax(np.nan_to_num(speed[spont:spont+frames_post])) + spont
                    plot_traces(spont_homing_speed_axs, shelter_distance, speed, ang_vel, rsig, speed_onset, magenta)
                    spont_speed_traces[r].append(rsig[speed_onset-frames_pre:speed_onset+frames_post])

                # Plot spont runs
                for spont in spont_runs:
                    if spont < frames_pre + 1: continue
                    if shelter_distance[spont + frames_post + frames_pre] < 450: continue
                    plot_traces(spont_runs_axs, shelter_distance, speed, ang_vel, rsig, spont, orange)
                    spont_runs_traces[r].append(rsig[spont-frames_pre:spont+frames_post])


                if DEBUG: break

        # Now plot averages
        for r, roi_id in enumerate(roi_ids):
            f, (stim_aligned_axs, escape_aligned_axs, speed_aligned_axs, shelt_aligned_axs, spont_homing_axs, spont_homing_speed_axs, spont_runs_axs) = figures[r], axes[r]

            # Plt mean for spont homings
            if spont_traces[r]:
                plot_roi_mean_sig(spont_homing_axs[4], spont_traces[r], random_traces[r], label=f"{len(spont_traces[r])} spont. homings")
                spont_homing_axs[4].legend()

                plot_roi_mean_sig(spont_homing_speed_axs[4], spont_speed_traces[r], random_traces[r], label=f"{len(spont_speed_traces[r])} spont. homings")
                spont_homing_speed_axs[4].legend()

            # Plt mean for spont runs
            if spont_runs_traces[r]:
                plot_roi_mean_sig(spont_runs_axs[4], spont_runs_traces[r], random_traces[r], label=f"{len(spont_runs_traces[r])} spont. runs")
                spont_runs_axs[4].legend()


            # Plot mean ROI traces
            if n_trials[r]:
                plot_roi_mean_sig(stim_aligned_axs[4], stim_traces[r], random_traces[r])
                plot_roi_mean_sig(escape_aligned_axs[4], escape_traces[r], random_traces[r])
                plot_roi_mean_sig(shelt_aligned_axs[4], shelt_traces[r], random_traces[r])
                plot_roi_mean_sig(speed_aligned_axs[4], speed_traces[r], random_traces[r])

            if DEBUG: break

        # Now Fix axes and save figures
        for r, roi_id in enumerate(roi_ids):
            f, (stim_aligned_axs, escape_aligned_axs, speed_aligned_axs, shelt_aligned_axs, spont_homing_axs, spont_homing_speed_axs, spont_runs_axs) = figures[r], axes[r]
            # plot_roi_correlation(whole_traces[roi_id], whole_traces['speed'], corr_axs[1])
            # plot_roi_correlation(whole_traces[roi_id], whole_traces['shelt_dist'], corr_axs[0])

            # Final fixes in figure
            f.suptitle(f"{rec['mouse']} - {sess} - {roi_id} - {n_trials[r]} trials")

            for ax in f.axes: 
                adjust_ticks(ax, frames_pre, frames_post, behav_fps, every=1)
                ax.set(xlim=[xmin, xmax])

        

            if DEBUG: break
            else:
                save_figure(f, paths[r], verbose=True)                
                plt.close(f)
            if DEBUG: break
        if DEBUG: break
    if DEBUG: break


        # PLOT WHOLE SESSION PCA
        # if not DEBUG:
        #     means = []
        #     for trace in escape_traces:
        #         means.append(np.mean(trace, axis=0))
        #     means = np.vstack(means)
        #     f, ax = plt.subplots(figsize=(12,12))

        #     pca = PCA(n_components=2)
        #     PCs = pca.fit_transform(means.T)
        #     ax.scatter(PCs[:frames_pre, 0], PCs[:frames_pre, 1], color='k', s=15,  zorder=99, edgecolors=[.2, .2, .2], lw=1)
        #     ax.scatter(PCs[frames_pre:, 0], PCs[frames_pre:, 1], color='r', s=15,  zorder=99, edgecolors=[.2, .2, .2], lw=1)
        #     ax.scatter(PCs[frames_pre, 0], PCs[frames_pre, 1], color=royalblue, s=90,  zorder=99, edgecolors=[.2, .2, .2], lw=1)
        #     ax.plot(PCs[:, 0], PCs[:, 1], color='k', lw=1)
            

plt.show()


