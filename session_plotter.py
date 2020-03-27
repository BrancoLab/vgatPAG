"""
    Plots an overview of all trials in a session, showing botht he behaviour and the signals 
    from the ROIs

"""
import sys
sys.path.append("./")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from fcutils.plotting.colors import colorMap, desaturate_color
from fcutils.plotting.colors import *
from fcutils.plotting.utils import create_figure, clean_axes, save_figure, set_figure_subplots_aspect
from fcutils.objects import sort_list_of_strings
from fcutils.file_io.utils import check_create_folder

from vgatPAG.database.db_tables import Tracking, AudioStimuli, VisualStimuli, Session, Roi, Recording
from vgatPAG.paths import output_fld
from vgatPAG.variables import miniscope_fps, tc_colors


# ----------------------------------- Setup ---------------------------------- #
# ? Define some params
n_sec_pre = 8 # keep it a multiple of 4 please
n_sec_post = 20 # keep it a multiple of 4 please

ONLY_ESCAPES = False
INDIVIDUAL_TRAILS = True

# Get all sessions
sessions = Session.fetch("sess_name")

# Get the recordings for each session
recordings = {s:(Recording & f"sess_name='{s}'").fetch(as_dict=True) for s in sessions}



def figure_generator(nrois):
    """
        Generate the figure layout
    """
    nrois = len(roi_ids)
    ncols = int(np.ceil(nrois/2))
    nrows = 6
        
    f = plt.figure(figsize=(30, 15))

    gs = f.add_gridspec(nrows, ncols)

    tracking_ax = f.add_subplot(gs[:2, :])
    tracking_ax.set(title="Tracking", xticks=[], yticks=[])

    speed_ax = f.add_subplot(gs[2, :])
    speed_ax.set(title="Speed", ylabel="speed, px/frame")

    dist_ax = f.add_subplot(gs[3, :])
    dist_ax.set(title="Shelter distance", ylabel='shelter distance in px', xlabel="frames")

    roi_axes = []
    for n in range(2):
        for i in range(ncols):
            if n == 0 and i == 0:
                ax = f.add_subplot(gs[4+n, i])
            else:
                ax = f.add_subplot(gs[4+n, i], sharex=roi_axes[0])
            ax.set(title=f"ROI_{(n+1)*(i+1)}")
            ax.margins(0.1, 1)
            roi_axes.append(ax)

    set_figure_subplots_aspect(left=0.05, right=0.95, bottom=0.05, top=0.96, wspace=0.3, hspace=0.6)
    clean_axes(f)
    return f, tracking_ax, speed_ax, dist_ax, roi_axes


def check_if_escape(x_track):
    if np.any(np.where(x_track < 300)[0]):
        return True
    else: 
        return False

def adjust_ticks(ax, pre_frames, post_frames, fps, ervery=2):
    xticks = np.arange(0, pre_frames, ervery*fps)
    xticks_labels = [int(pre_frames/fps - x/fps) for x in xticks]

    xticks2 = np.arange(pre_frames, post_frames+pre_frames, ervery*fps)
    xticks_labels2 = [int(x/fps - pre_frames/fps) for x in xticks2]
    ax.set(xticks=np.concatenate([xticks, xticks2]), xticklabels=np.concatenate([xticks_labels, xticks_labels2]))


# ----------------------------- Generate figures ----------------------------- #
# Loop over sessions
for sess in tqdm(sessions):
    # ----------------------------- Get sessiond data ---------------------------- #
    for rec in  recordings[sess]:
        behav_fps = rec['fps_behav']
        frames_pre_behav = n_sec_pre * behav_fps
        frames_post_behav = n_sec_post * behav_fps
        frames_pre_ca = n_sec_pre * miniscope_fps
        frames_post_ca = n_sec_post * miniscope_fps


        # Get stimuli
        vstims = (VisualStimuli & f"sess_name='{sess}'" & f"rec_name='{rec['rec_name']}'").fetch("frame")
        astims = (AudioStimuli & f"sess_name='{sess}'" & f"rec_name='{rec['rec_name']}'").fetch("frame")

        vstim_colors = [colorMap(i, name='Reds', vmin=-12, vmax=len(vstims)+6) for i in range(len(vstims))]
        astim_colors = [colorMap(i, name='Greens', vmin=-12, vmax=len(astims)+6) for i in range(len(astims))]
    
        # Get tracking
        body_tracking = np.vstack((Tracking * Tracking.BodyPartTracking & f"sess_name='{sess}'"
                                    & f"rec_name='{rec['rec_name']}'" & "bp='body'").fetch1("x", "y", "speed"))

        snout_tracking = np.vstack((Tracking * Tracking.BodyPartTracking & f"sess_name='{sess}'"
                                    & f"rec_name='{rec['rec_name']}'" & "bp='snout'").fetch1("x", "y", "speed"))

        tail_tracking = np.vstack((Tracking * Tracking.BodyPartTracking & f"sess_name='{sess}'"
                                    & f"rec_name='{rec['rec_name']}'" & "bp='tail_base'").fetch1("x", "y", "speed"))

        # Get ROIs
        roi_ids, roi_sigs = (Roi & f"sess_name='{sess}'" & f"rec_name='{rec['rec_name']}'").fetch("roi_id", "signal", as_dict=False)
        
        
        # Create figure
        nrois = len(roi_ids)
        f, tracking_ax, speed_ax, dist_ax, roi_axes = figure_generator(nrois)

        # -------------------------------- Plot stuff -------------------------------- #
        plotted_vline = False
        for st, (stims, colors) in enumerate(zip((vstims, astims), (vstim_colors, astim_colors))):
            for s, (stim, color) in enumerate(zip(stims, colors)):
                if INDIVIDUAL_TRAILS:
                    f1 = os.path.join(output_fld, f"{rec['mouse']}")
                    check_create_folder(f1)
                    fld = os.path.join(f1, rec['mouse'])
                    check_create_folder(fld)
                    f, tracking_ax, speed_ax, dist_ax, roi_axes = figure_generator(nrois)
                    plotted_vline = False



                # Plot tracking
                btrack = body_tracking[:, stim-frames_pre_behav:stim+frames_post_behav]
                strack = snout_tracking[:, stim-frames_pre_behav:stim+frames_post_behav]
                ttrack = tail_tracking[:, stim-frames_pre_behav:stim+frames_post_behav]

                if ONLY_ESCAPES and not check_if_escape(btrack[0, frames_pre_behav:]):
                    continue
                
                tracking_ax.scatter(btrack[0, frames_pre_behav:], btrack[1, frames_pre_behav:], 
                                            color=color, s=15, alpha=.6,  zorder=80, edgecolors=[.2, .2, .2], lw=.2)
                tracking_ax.scatter(btrack[0, frames_pre_behav], btrack[1, frames_pre_behav], 
                                            color=royalblue, s=80, alpha=1,  zorder=99, edgecolors='k', lw=1)

                if INDIVIDUAL_TRAILS:
                    tracking_ax.scatter(strack[0, frames_pre_behav:], strack[1, frames_pre_behav:], 
                                                color=color, s=30, alpha=.6,  zorder=150, edgecolors=[.2, .2, .2], lw=1)
                    tracking_ax.plot([btrack[0, frames_pre_behav::2], strack[0, frames_pre_behav::2]], 
                                    [btrack[1, frames_pre_behav::2], strack[1, frames_pre_behav::2]],
                                    color = desaturate_color(color), alpha=.5)
                    tracking_ax.set(xlim=[200, 1100], ylim=[0, 500])

                tracking_ax.plot(btrack[0, frames_pre_behav:], btrack[1, frames_pre_behav:], color=desaturate_color(color), alpha=.8, lw=.4)
                
                # Plot speed and distance
                speed_ax.plot(btrack[2, :], color=color, alpha=.5)           
                dist_ax.plot(btrack[0, :], color=color, alpha=.5)

                if not plotted_vline:
                    dist_ax.axvline(frames_pre_behav, lw=2, color=[.4, .4, .4], ls="--", zorder=99)
                    speed_ax.axvline(frames_pre_behav, lw=2, color=[.4, .4, .4], ls="--", zorder=99)

                # plot rois
                for ax, rid, rsig in zip(roi_axes, roi_ids, roi_sigs):
                    ax.plot(rsig[stim-frames_pre_ca:stim+frames_post_ca], color=color, alpha=1)

                    if not plotted_vline:
                        ax.axvline(frames_pre_ca, lw=2, color=[.4, .4, .4], ls="--", zorder=99)
                plotted_vline = True
                
                if INDIVIDUAL_TRAILS:
                    # Refine some stuff x ticks
                    for ax in [speed_ax, dist_ax]:
                        adjust_ticks(ax, frames_pre_behav, frames_post_behav, behav_fps)

                    for ax in roi_axes:
                        adjust_ticks(ax, frames_pre_ca, frames_post_ca, miniscope_fps, ervery=4)
                        
                    if st == 0:
                        stp = 'visual'
                    else: 
                        stp = 'audio'
                    save_figure(f, os.path.join(fld, f"{rec['mouse']}_{sess}_{stp}_trial{s}"), verbose=False)
                    plt.close(f)

        # Refine some stuff x ticks
        for ax in [speed_ax, dist_ax]:
            adjust_ticks(ax, frames_pre_behav, frames_post_behav, behav_fps)

        for ax in roi_axes:
            adjust_ticks(ax, frames_pre_ca, frames_post_ca, miniscope_fps, ervery=4)

    if not INDIVIDUAL_TRAILS:
        save_figure(f, os.path.join(output_fld, f"{rec['mouse']}_{sess}_summary"))
        plt.close(f)
