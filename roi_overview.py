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


from vgatPAG.database.db_tables import Trackings, Session, Roi, Recording, Mouse, Event
from vgatPAG.paths import output_fld
from vgatPAG.variables import miniscope_fps, tc_colors, shelter_width_px, stims_colors, spont_events_colors


np.warnings.filterwarnings('ignore')



# ----------------------------------- Setup ---------------------------------- #
# ? Define some params
n_sec_pre = 2 
n_sec_post = 8
n_sec_post_plot = 5

DEBUG = True

mice = Mouse.fetch("mouse")

# Get all sessions
sessions = {m:(Session & f"mouse='{m}'").fetch("sess_name") for m in mice}

# Get the recordings for each session
recordings = {m:{s:(Recording & f"sess_name='{s}'" & f"mouse='{m}'").fetch(as_dict=True) for s in sessions[m]} for m in mice}





# ------------------------------- Figure utils ------------------------------- #
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

# ------------------------------ Plotting Utils ------------------------------ #
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
    nframes = np.int(fps * .5)
    pre = np.nanmean(mean[frames_pre - nframes:frames_pre])
    post = np.nanmean(mean[frames_pre:frames_pre + nframes])

    pre_vs_post = (post-pre) / (pre+post)
    ax.set(title=r"$\frac{post-pre}{pre+post} = "+ str(round(pre_vs_post, 2)) +"$")
    y = ax.get_ylim()[0]+2
    if pre_vs_post < 0:
        rec = patches.Rectangle((y, frames_pre + (fps*pre_vs_post)), width=np.abs(fps*pre_vs_post), height=5,
                    facecolor=blackboard, alpha=.8)
    else:
        rec = patches.Rectangle((y, frames_pre), width=np.abs(fps*pre_vs_post), height=5,
                    facecolor=blackboard, alpha=.8)
    ax.add_artist(rec)

    if label is not None:
        ax.legend()




# ---------------------------------------------------------------------------- #
#                                  MAKE FIGUES                                 #
# ---------------------------------------------------------------------------- #
# with plt.xkcd(): # ! use this for funzies

# Loop over sessions
for mouse in mice:
    for sess in sessions[mouse]:
        # Fetch some data
        recs = Recording().get_sessions_recordings(sess)
        roi_ids, roi_sigs, nrois = Roi().get_sessions_rois(sess)
        roi_ids = list(roi_ids.values())[0]
        evoked, spont = Event().get_sessions_events(sess)

        # prep some variables
        fps = Recording().get_recording_fps(rec_name=recs[0])
        frames_pre = n_sec_pre * fps
        frames_post = n_sec_post * fps
        xmax = n_sec_post_plot * fps
        xmin = 0

        # Loop over each ROI
        print(f"Mouse {mouse} - session: {sess} - [{len(roi_ids)} rois]")
        for n in tqdm(range(list(nrois.values())[0])):
            # Create a figure for the ROI
            f, axs = create_figure(xmin, xmax)            
            axes = dict(
                stim_onset = axs[0],
                escape_onset=axs[1],
                escape_peak_speed=axs[2],
                shelter_arrival=axs[3],
                homing=axs[4],
                homing_peak_speed=axs[5],
                outrun=axs[6],
            )
            
            f_path = get_figure_path(mouse, sess, roi_ids[n])

            # traces: keep ROI signal data aligned to various events to plot averages
            traces = dict(
                stim_onset=[],
                escape_onset=[],
                escape_peak_speed=[],
                shelter_arrival=[],
                homing=[],
                homing_peak_speed=[],
                outrun=[],
                random=[],
            )

            # Loop over each recording in the session
            tot_trials = 0
            for rec in recs:
                rsig = roi_sigs[rec][n]
                evkd = evoked[rec]
                spnt = spont[rec]

                # Get some more stuff
                body_tracking, ang_vel, speed, shelter_distance = Trackings().get_recording_tracking_clean(sess_name=sess, rec_name=rec)

                # Plot evoked events
                for i, ev in evkd.iterrows():
                    axs = axes[ev.type]
                    color = stims_colors[ev.stim_type]
                    plot_traces(axs, shelter_distance, speed, ang_vel, rsig, ev.frame, color)
                    traces[ev.type].append(rsig[ev.frame-frames_pre:ev.frame+frames_post])

                # Plot spont events
                for i, ev in spnt.iterrows():
                    axs = axes[ev.type]
                    color = spont_events_colors[ev.type]
                    plot_traces(axs, shelter_distance, speed, ang_vel, rsig, ev.frame, color)
                    traces[ev.type].append(rsig[ev.frame-frames_pre:ev.frame+frames_post])

                # Get ROI random sample
                n_trials = (len(evkd.loc[evkd.type=="stim_onset"]))
                tot_trials += n_trials
                for i in range(n_trials):
                    traces['random'].append(list(Roi().get_roi_signal_at_random_time(rec, roi_ids[n], frames_pre, frames_post)))

            # Plot mean roi signal traces
            for key, trace in traces.items():
                if key == 'random': continue
                plot_roi_mean_sig(axes[key][4], trace, traces['random'])


            # Refine figure
            f.suptitle(f"{mouse} - {sess} - {roi_ids[n]} - {tot_trials} trials")
            for ax in f.axes: 
                adjust_ticks(ax, frames_pre, frames_post, fps, every=1)
                ax.set(xlim=[xmin, xmax])

            if DEBUG: break
        if DEBUG: break
    if DEBUG: break
plt.show()

# TODO exclude spont events that include no roi signal


