import sys
sys.path.append("./")

import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import matplotlib.patches as patches
from scipy.stats import sem

from fcutils.plotting.colors import desaturate_color
from fcutils.plotting.colors import *
from fcutils.plotting.utils import clean_axes, save_figure, set_figure_subplots_aspect
from fcutils.file_io.utils import check_create_folder


from vgatPAG.database.db_tables import Trackings, Session, Roi, Recording, Mouse
from vgatPAG.paths import output_fld
from vgatPAG.variables import miniscope_fps, tc_colors, shelter_width_px


np.warnings.filterwarnings('ignore')



# ----------------------------------- Setup ---------------------------------- #
# ? Define some params
n_sec_pre = 2 
n_sec_post = 8
n_sec_post_plot = 5
n_sec_spont_events = 2

DEBUG = False

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
    ax.axvspan(frames_pre-(fps * .5), frames_pre+(fps * .5), facecolor=desaturate_color(deepskyblue), alpha=0.075, zorder=-1)


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
    nframes = np.int(behav_fps * .5)
    pre = np.nanmean(mean[frames_pre - nframes:frames_pre])
    post = np.nanmean(mean[frames_pre:frames_pre + nframes])

    pre_vs_post = (post-pre) / (pre+post)
    ax.set(title=r"$\frac{post-pre}{pre+post} = "+ str(round(pre_vs_post, 2)) +"$")
    y = ax.get_ylim()[0]+2
    if pre_vs_post < 0:
        rec = patches.Rectangle((y, frames_pre + (behav_fps*pre_vs_post)), width=np.abs(behav_fps*pre_vs_post), height=5,
                    facecolor=blackboard, alpha=.8)
    else:
        rec = patches.Rectangle((y, frames_pre), width=np.abs(behav_fps*pre_vs_post), height=5,
                    facecolor=blackboard, alpha=.8)
    ax.add_artist(rec)




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
            roi_ids, roi_sigs, nrois = Roi().get_recordings_rois(sess_name=sess, rec_name=rec)
            # roi_ids, roi_sigs = (Roi & f"sess_name='{sess}'" & f"rec_name='{rec['rec_name']}'").fetch("roi_id", "signal", as_dict=False)
            # nrois.append(len(roi_ids))
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

            # TODO clener get stimuli, tracking and spont events
            roi_ids, roi_sigs, nrois = Roi().get_recordings_rois(sess_name=sess, rec_name=rec)


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

          

plt.show()


