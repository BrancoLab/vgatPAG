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
from sklearn.decomposition import PCA
import matplotlib.patches as patches
from scipy.stats import zscore

from fcutils.plotting.colors import colorMap, desaturate_color
from fcutils.plotting.colors import *
from fcutils.plotting.utils import create_figure, clean_axes, save_figure, set_figure_subplots_aspect, forceAspect
from fcutils.plotting.plot_elements import plot_shaded_withline
from fcutils.objects import sort_list_of_strings
from fcutils.file_io.utils import check_create_folder

from vgatPAG.database.db_tables import Tracking, AudioStimuli, VisualStimuli, Session, Roi, Recording
from vgatPAG.paths import output_fld
from vgatPAG.variables import miniscope_fps, tc_colors, shelter_width_px


# ----------------------------------- Setup ---------------------------------- #
# ? Define some params
n_sec_pre = 12 # keep it a multiple of 4 please
n_sec_post = 24 # keep it a multiple of 4 please

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
    ncols = 8
    nrows = 7
        
    f = plt.figure(figsize=(25, 12))

    gs = f.add_gridspec(nrows, ncols)

    tracking_ax = f.add_subplot(gs[:2, :2])
    tracking_ax.set(title="Tracking", xticks=[], yticks=[], xlim=[200, 1100], ylim=[0, 500])

    speed_ax = f.add_subplot(gs[2, :2])
    speed_ax.set(title="Speed", ylabel="speed, px/frame")

    dist_ax = f.add_subplot(gs[3, :2])
    dist_ax.set(title="Shelter distance", ylabel='shelter distance in px', ylim=[-200, 1000])

    roi_ax = f.add_subplot(gs[4, :2])
    roi_ax.set(title="ROI traces", ylabel='Activity', xlabel="seconds")

    pca_ax = f.add_subplot(gs[5, 0], aspect="equal")
    pca_ax.set(title="PCA trace", ylabel='PC1', xlabel="PC2")

    heatmap_ax = f.add_subplot(gs[5, 1])
    heatmap_ax.set(title="z scored activity", ylabel='ROI', xlabel="seconds")
    
    speedcorr_ax = f.add_subplot(gs[6, 0])
    speedcorr_ax.set(title="Speed correlation", ylabel='ROI', xlabel="seconds")
    
    distcorr_ax = f.add_subplot(gs[6, 1])
    distcorr_ax.set(title="Shelter dist correlation", ylabel='ROI', xlabel="seconds")

    counter = 0
    row = 0
    roi_axes = []
    for rn in range(nrois):
        if counter == ncols-2:
            counter = 0
            row += 1
        counter += 1
        roi_axes.append(f.add_subplot(gs[row, counter+1]))
        roi_axes[-1].set(title=f"ROI {rn}")


    set_figure_subplots_aspect(left=0.05, right=0.95, bottom=0.05, top=0.93, wspace=0.3, hspace=0.8)
    clean_axes(f)
    return f, tracking_ax, speed_ax, dist_ax, roi_ax, roi_axes, pca_ax, heatmap_ax, speedcorr_ax, distcorr_ax


def check_if_escape(x_track):
    if np.any(np.where(x_track < 300)[0]):
        return True
    else: 
        return False

def adjust_ticks(ax, pre_frames, post_frames, fps, every=2):
    xticks = np.arange(0, pre_frames, every*fps)
    xticks_labels = [-int(pre_frames/fps - x/fps) for x in xticks]

    xticks2 = np.arange(pre_frames, post_frames+pre_frames, every*fps)
    xticks_labels2 = [int(x/fps - pre_frames/fps) for x in xticks2]
    ax.set(xticks=np.concatenate([xticks, xticks2]), xticklabels=np.concatenate([xticks_labels, xticks_labels2]))


def get_other_stims_in_window(vstims, astims, stim, preframes, postframes):
    vclose = [preframes + v-stim for v in vstims if np.abs(stim - v) < preframes+postframes and v < stim]
    aclose = [preframes + a-stim for a in astims if np.abs(stim - a) < preframes+postframes and a < stim]
    return vclose, aclose

def get_rolling_cor(roi_sigs, corr_sig, window=20, center=True):
    sigs = pd.DataFrame(roi_sigs.T).fillna(method='bfill')
    corr = sigs.rolling(window, center=center).corr(other=pd.Series(corr_sig))
    return corr.values.T

# ----------------------------- Generate figures ----------------------------- #
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
        roi_sigs = [list(r) for r in roi_sigs]
        nrois = len(roi_ids)
        roi_colors = [colorMap(i, name='tab20', vmin=-4, vmax=nrois+2) for i in range(nrois)]

        # -------------------------------- Plot stuff -------------------------------- #
        plotted_vline = False
        for st, (stype, stims, colors) in enumerate(zip(['visual', 'audio'], (vstims, astims), (vstim_colors, astim_colors))):
            for s, (stim, color) in enumerate(zip(stims, colors)):
                # Create figure and get save path
                f1 = os.path.join(output_fld, f"{rec['mouse']}")
                check_create_folder(f1)
                fld = os.path.join(f1, sess)
                check_create_folder(fld)
                f, tracking_ax, speed_ax, dist_ax, roi_ax, roi_axes, pca_ax, heatmap_ax, speedcorr_ax, distcorr_ax\
                            = figure_generator(nrois)
                f.suptitle(f"{rec['mouse']} - {sess} ~ {stype}, frame {stim}")

                # Plot tracking
                btrack = body_tracking[:, stim-frames_pre:stim+frames_post]
                strack = snout_tracking[:, stim-frames_pre:stim+frames_post]
                ttrack = tail_tracking[:, stim-frames_pre:stim+frames_post]
                shelter_distance = btrack[0, :]-shelter_width_px-200

                try:
                    at_shelter = np.where(shelter_distance[frames_pre:] <= 0 )[0][0]
                    at_shelter += frames_pre
                except:
                    at_shelter = None

                rect = patches.Rectangle((200, 0), shelter_width_px, 500,
                                        linewidth=1,edgecolor='k',facecolor='k', alpha=.25)
                tracking_ax.add_patch(rect)

                if ONLY_ESCAPES and not check_if_escape(btrack[0, frames_pre:]):
                    continue
                
                tracking_ax.scatter(btrack[0, frames_pre:], btrack[1, frames_pre:], 
                                            color=color, s=15, alpha=.6,  zorder=80, edgecolors=[.2, .2, .2], lw=.2)
                tracking_ax.scatter(btrack[0, frames_pre], btrack[1, frames_pre], 
                                            color=royalblue, s=80, alpha=1,  zorder=99, edgecolors='k', lw=1)

                tracking_ax.scatter(strack[0, frames_pre:], strack[1, frames_pre:], 
                                            color=color, s=30, alpha=.6,  zorder=150, edgecolors=[.2, .2, .2], lw=1)
                tracking_ax.plot([btrack[0, frames_pre::2], strack[0, frames_pre::2]], 
                                [btrack[1, frames_pre::2], strack[1, frames_pre::2]],
                                color = desaturate_color(color), alpha=.5)

                tracking_ax.plot(btrack[0, frames_pre:], btrack[1, frames_pre:], color=desaturate_color(color), alpha=.8, lw=.4)
                
                # Plot speed and distance
                X = np.arange(btrack.shape[1])
                plot_shaded_withline(speed_ax, X, btrack[2, :], color=color)
                plot_shaded_withline(dist_ax, X, shelter_distance, color=color)
                if at_shelter is not None:
                    for ax in [speed_ax, dist_ax]:
                        ax.axvline(at_shelter, lw=2, ls=":", color=[.2, .2, .2])


                # Add vline where stimuli happened
                vclose, aclose = get_other_stims_in_window(vstims, astims, stim,frames_pre, frames_post)
                for slist in [vclose, aclose, [frames_pre]]:
                    for ax in [dist_ax, speed_ax]:
                        for st in slist:
                            ax.axvline(st, lw=2, color=[.4, .4, .4], ls="--", zorder=99)

                # plot rois
                for rid, rsig, rcol, ax in zip(roi_ids, roi_sigs, roi_colors, roi_axes):
                    roi_ax.plot(rsig[stim-frames_pre:stim+frames_post], color=rcol, alpha=1, label=rid)
                    plot_shaded_withline(ax, np.arange(frames_pre+frames_post), 
                                rsig[stim-frames_pre:stim+frames_post], color=desaturate_color(color))

                # Vertical line at relevant times
                for slist in [[frames_pre]]:
                        for st in slist:
                            roi_ax.axvline(st, lw=2, color=[.4, .4, .4], ls="--", zorder=99)
                            for ax in roi_axes:
                                ax.axvline(st, lw=2, color=[.4, .4, .4], ls="--", zorder=99)
                                if at_shelter is not None:
                                    ax.axvline(at_shelter, lw=2, ls=":", color=[.2, .2, .2])

                # Get sorted ROI traces for this trial
                rois_signal = np.array(roi_sigs)[:, stim-frames_pre:stim+frames_post]
                sorted_rois_signal = np.array([roi_sigs[i] for i in np.argsort(np.argmax(rois_signal, axis=1))])[:, stim-frames_pre:stim+frames_post]

                # Plot trace in PCA space
                pca = PCA(n_components=2)
                PCs = pca.fit_transform(rois_signal.T)
                pca_ax.scatter(PCs[:, 0], PCs[:, 1], c=np.arange(PCs.shape[0]), s=10, cmap='Purples', zorder=99, edgecolors=[.2, .2, .2], lw=1)
                pca_ax.plot(PCs[:, 0], PCs[:, 1], color='k', lw=1)
                pca_ax.scatter(PCs[0, 0], PCs[0, 1], color=royalblue, s=80, alpha=1,  zorder=99, edgecolors='k', lw=1)
                
                # Plot traces as heatmap
                heatmap_ax.imshow(zscore(sorted_rois_signal, axis=1), origin="upper", cmap='bone')
                forceAspect(heatmap_ax, aspect=1)

                # Get speed and distance correlations
                speed_corr = get_rolling_cor(rois_signal, btrack[2, :])
                dist_corr = get_rolling_cor(rois_signal, shelter_distance)

                speedcorr_ax.imshow(speed_corr, origin="upper", cmap='bwr', vmin=-1, vmax=1)
                forceAspect(speedcorr_ax, aspect=1)

                distcorr_ax.imshow(dist_corr, origin="upper", cmap='bwr', vmin=-1, vmax=1)
                forceAspect(distcorr_ax, aspect=1)

                # Refine some stuff x ticks
                for ax in [speed_ax, dist_ax]:
                    adjust_ticks(ax, frames_pre, frames_post, behav_fps)

                adjust_ticks(roi_ax, frames_pre, frames_post, behav_fps)

                for ax in [heatmap_ax, distcorr_ax, speedcorr_ax]:
                    adjust_ticks(ax, frames_pre, frames_post, behav_fps, every=6)
                    ax.axvline(frames_pre, lw=2, color='k')

                for ax in roi_axes:
                    adjust_ticks(ax, frames_pre, frames_post, behav_fps, every=4)
                    ax.set(ylim=roi_ax.get_ylim())
                    
                if st == 0:
                    stp = 'visual'
                else: 
                    stp = 'audio'
                save_figure(f, os.path.join(fld, f"{rec['mouse']}_{sess}_{stp}_trial{s}"), verbose=False)

                plt.show()
                plt.close(f)

                a = 1

