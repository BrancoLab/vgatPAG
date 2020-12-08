import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from rich.progress import track
import pandas as pd
from myterial import cyan, teal, indigo, orange, salmon, grey_light, grey, grey_dark, blue_grey_darker
from myterial import brown_dark, grey, blue_grey_dark, grey_darker, salmon_darker, orange_darker
from scipy.stats import zscore
from collections import namedtuple
from pyrnn.analysis.dimensionality import get_n_components_with_pca
from brainrender._colors import map_color

from pyinspect import install_traceback
install_traceback()

from fcutils.plotting.utils import calc_nrows_ncols, clean_axes, save_figure
from fcutils.maths.utils import rolling_mean, derivative
from vgatPAG.database.db_tables import ManualBehaviourTags, Roi, Sessions

from Analysis import get_session_data, signal_color, get_session_tags, speed_color, shelt_dist_color

# %%

pre_pos_s = 1.5
M = int(pre_pos_s*30)
lbls = ('stim', 'start', 'run', 'shelter', 'stop')

xlbl = dict(
        xlabel='time from tag\n(s)', 
        xticks=[0, M, 2*M], 
        xticklabels=[-pre_pos_s, 0, pre_pos_s]
        )


fld = Path('D:\\Dropbox (UCL)\\Project_vgatPAG\\analysis\\doric\\Fede\\\ddf_tag_aligned_all_tags_V2')

# %%
def dff(sig, n_frames_pre):
    th = np.nanpercentile(sig[:n_frames_pre], .3)
    return rolling_mean((sig - th)/th, 3), th

def get_tags_sequences(tags):
    sequence = namedtuple('sequence', 'STIM, H, B, C, E')
    sequences = []
    
    # get stimuli
    stims = tags.session_stim_frame.unique()

    for n, stim in enumerate(stims):
        seq = [stim]
        tgs = tags.loc[(tags.session_stim_frame == stim)&(tags.session_frame>=stim)]
        if n < len(stims)-1:
            tgs = tgs.loc[tgs.session_frame < stims[n+1]]

        for ttype in ('H','B','C','E'):
            nxt = tgs.loc[(tgs.tag_type==f'VideoTag_{ttype}')]

            if nxt.empty:
                seq.append(None)
            else:
                if nxt.session_frame.values[0] - stim > 15*30:
                    seq.append(None)
                else:
                    seq.append(nxt.session_frame.values[0])
        sequences.append(sequence(*seq))
    return sequences

def get_active_rois(rois, sequences):
    active = []
    for roi in rois.columns:
        for seq in sequences:
            roi_mean = np.nanmean(rois[roi][seq.STIM - 15*30:seq.STIM])
            roi_std = np.nanstd(rois[roi][seq.STIM - 15*30:seq.STIM])

            if (np.any(rois[roi][seq.STIM:seq.E] > (roi_mean + 2*roi_std)) 
                        or np.any(rois[roi][seq.STIM:seq.E] < (roi_mean - 2*roi_std))):
                active.append(roi)
                break
    print(f'{len(active)} rois out of {len(rois.columns)} were considered active')
    active_rois = rois[active]
    return active_rois

def get_PCs(active_rois):
    active_rois[data.is_rec==0] = 0
    R = active_rois.values[data.is_rec==1].astype(np.float64)
    scaler = StandardScaler().fit(R)

    pca = PCA(n_components=3).fit(scaler.transform(R))

    pcs = pca.transform(scaler.transform(active_rois.values))
    
    active_rois[data.is_rec==0] = np.nan
    mean_activity = np.nanmean(active_rois.values, 1)
    return pcs, mean_activity, active_rois

# %%
for sess in Sessions.fetch(as_dict=True):
    print(sess['mouse'], sess['date'])
    data, rois = get_session_data(sess['mouse'], sess['date'], roi_data_type='raw')
    if data is None: continue

    tags = get_session_tags(sess['mouse'], sess['date'], 
                        etypes=('visual', 'audio', 'audio_visual'), 
                        ttypes=('H', 'B', 'C', 'E'))
    
    # get tags sequences
    sequences = get_tags_sequences(tags)
    colors = [map_color(n, 'viridis', 0, len(sequences)) for n in range(len(sequences))]

    # Get which ROIs show escape-related acivity
    active_rois = get_active_rois(rois, sequences)

    # fit PCA to whole session recording
    try:
        pcs, mean_activity, active_rois = get_PCs(active_rois)
    except  ValueError:
        continue

    # Loop over ROIs
    for roi in rois.columns:
        f, axarr = plt.subplots(ncols=5, nrows=3, figsize=(16, 9), sharex=True)
        f.suptitle('ACTIVE ROI' if roi in active_rois.columns else 'NOT active ROI')
        
        # loop over sequences
        prev_stim = 0
        for color, seq in zip(colors, sequences):
            if seq.E is None or seq.C is None or seq.B is None or seq.H is None: continue
            if seq.STIM - prev_stim < 60*30:
                continue
            else:
                pre_stim = seq.STIM

            start = seq.STIM - 5 * 30
            end = seq.E + 5 * 30
            if data.is_rec[start] == 0: continue  # stim when not recording
            
            for n, frame in enumerate(seq):
                if frame is None: continue

                # Plot first PC
                axarr[1, n].plot(pcs[frame-M:frame+M, 0], lw=3, color=color)
                # ax0s[n].plot(mean_activity[frame-M:frame+M], lw=3, color='salmon')

                # Plot roi trace
                sig, th = dff(rois[roi][start:end], 5*30)
                rel_frame = frame - seq.STIM + 5 * 30
                axarr[2, n].plot(sig[rel_frame-M:rel_frame+M], lw=3, color=color)

                # Plot speed trace
                axarr[0, n].plot(data.s[frame-M:frame+M].values, lw=3, color=color)

        # Set figure
        axarr[0, 0].set(ylabel='Speed')
        # ax0s[0].set(ylabel='Mean FOV activity')
        axarr[1, 0].set(ylabel='First PC on RAW data\nPOPULATION ACTIVITY')
        axarr[2, 0].set(ylabel='DFF\nSINGLE ROI')

        for lbl, ax in zip(lbls, axarr[0, :]):
            ax.set(title = lbl)

        for ax in axarr[2, :]:
            ax.set(**xlbl)
        
        for ax in axarr.flatten():
            ax.axvline(M, lw=3, color=[.4, .4, .4], zorder=200)
            ax.axvspan(-2, M, color='w', alpha=.6, zorder=100)

        clean_axes(f)
        for row in range(3):
            maxy = np.max([ax.get_ylim()[1] for ax in axarr[row, :]])
            miny = np.min([ax.get_ylim()[0] for ax in axarr[row, :]])

            for n, ax in enumerate(axarr[row, :]):
                ax.set(ylim=[miny, maxy])
                if n > 0:
                    ax.spines['left'].set_visible(False)
                    ax.set(yticks=[])

                if row < 2:
                    ax.spines['bottom'].set_visible(False)


        save_figure(f, fld/f'{sess["mouse"]}_{sess["date"]}__{roi}', verbose=False)
        plt.close(f)
    del rois

        # break
    # break
# %%
# ''' Plot the mean trace aligned to each tag for each ROI and the first PCA component'''

# f, axarr = plt.subplots(ncols=5, nrows=2, figsize=(16, 9))

# for sessn, sess in enumerate(Sessions.fetch(as_dict=True)):
#     print(sess['mouse'], sess['date'])
#     data, rois = get_session_data(sess['mouse'], sess['date'], roi_data_type='raw')
#     if data is None: continue

#     tags = get_session_tags(sess['mouse'], sess['date'], 
#                         etypes=('visual', 'audio', 'audio_visual'), 
#                         ttypes=('H', 'B', 'C', 'E'))
    
#     # get tags sequences
#     sequences = get_tags_sequences(tags)

#     # Get which ROIs show escape-related acivity
#     active_rois = get_active_rois(rois, sequences)

#     # fit PCA to whole session recording
#     try:
#         pcs, mean_activity, active_rois = get_PCs(active_rois)
#     except  ValueError:
#         continue

#     # Loop over ROIs
#     for roi_n, roi in enumerate(rois.columns):
#         # loop over sequences
#         prev_stim = 0
#         chunks = {k:[] for k in ('STIM', 'H', 'B', 'C', 'E')}
#         for seq_n, seq in enumerate(sequences):
#             if seq.E is None or seq.C is None or seq.B is None or seq.H is None: continue
#             if seq.STIM - prev_stim < 60*30:
#                 continue
#             else:
#                 pre_stim = seq.STIM
#             if data.is_rec[seq.STIM] == 0: continue  # stim when not recording

#             start = seq.STIM - 5 * 30
#             end = seq.E + 5 * 30
#             sig, th = dff(rois[roi][start:end], 5 * 30)

#             for n, (tag, frame) in enumerate(seq._asdict().items()):
#                 if frame is None: continue
#                 rel_frame = frame - seq.STIM + 5 * 30
#                 chunks[tag].append(sig[rel_frame-M:rel_frame+M])

#                 if roi_n == 0 and seq_n==0:
#                     axarr[0, n].plot(pcs[frame-M:frame+M, 0], color=[.6, .6, .6], lw=3)



#         # Plot
#         roi_medians = {k:np.nanmedian(np.vstack(c), 0) for k, c in chunks.items()}
#         for (tag, chunk), ax in zip(roi_medians.items(), axarr[1, :]):
#             if np.max(chunk) > .8 or np.min(chunk) < -.8:

#                 color, alpha, lw = 'salmon', 1, 2
#             else:
#                 color, alpha, lw = [.5, .5, .5], .4, 1
#             ax.plot(chunk, color=color, alpha=alpha, lw=lw)

# # clean axes
# axarr[0, 0].set(ylabel='First PC')
# axarr[1, 0].set(ylabel='Rois average DFF')

# for ax in axarr.flatten():
#     ax.axvline(M, lw=3, color=[.4, .4, .4], zorder=200)
#     ax.axvspan(-2, M, color='w', alpha=.6, zorder=100)

# clean_axes(f)
# for row in range(2):
#     maxy = np.max([ax.get_ylim()[1] for ax in axarr[row, :]])
#     miny = np.min([ax.get_ylim()[0] for ax in axarr[row, :]])

#     for n, ax in enumerate(axarr[row, :]):
#         ax.set(ylim=[miny, maxy])
#         if n > 0:
#             ax.spines['left'].set_visible(False)
#             ax.set(yticks=[])

#         if row < 1:
#             ax.spines['bottom'].set_visible(False)
#             ax.set(xticks=[])



            

# %%
