# %%
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from rich.progress import track
import pandas as pd
from myterial import cyan, pink, grey_light, grey
from scipy.stats import zscore

from fcutils.plotting.utils import calc_nrows_ncols, clean_axes, save_figure
from fcutils.maths.filtering import smooth_hanning

from vgatPAG.database.db_tables import ManualBehaviourTags, Roi, Sessions
from Analysis import get_session_data, signal_color, get_session_tags, speed_color, shelt_dist_color

# %%
TAG = 'E'

pre_pos_s = 1.5
pre_pos = int(pre_pos_s*30)
n_s_pre, n_s_post = 5, 8
n_frames_pre = n_s_pre * 30
n_frames_post = n_s_post * 30
xlbl = dict(
        xlabel='time from tag\n(s)', 
        xlim=[0, n_frames_post + n_frames_pre],
        xticks=[0, n_frames_pre - pre_pos, n_frames_pre, n_frames_pre+pre_pos, n_frames_pre+n_frames_post], 
        xticklabels=[-n_s_pre, -pre_pos_s, 0, pre_pos_s, n_s_post]
        )


fld = Path('D:\\Dropbox (UCL)\\Project_vgatPAG\\analysis\\doric\\Fede\\ddf_tag_aligned')

def get_frames_numbers(event_frame, stim_frame):
        start, end = event_frame-n_frames_pre, event_frame+n_frames_post
        stim = n_frames_pre - (event_frame - stim_frame)
        if stim == n_frames_pre:
            stim = None
        return start, end, stim

def get_pre_vs_post(sig, overall_mean):
    pre = sig[n_frames_pre - pre_pos : n_frames_pre].mean()
    post = sig[n_frames_pre : n_frames_pre + pre_pos].mean()
    return pre, post

def mark_pre_post(ax):
    ax.axvspan(n_frames_pre - pre_pos, n_frames_pre, color=grey_light, zorder=-1)
    ax.axvspan(n_frames_pre, pre_pos + n_frames_pre, color=grey, zorder=-1)


def plot_heatmaps(ax, sig, tags):
    data = np.vstack([])


def dff(sig):
    th = np.nanpercentile(sig[:n_frames_pre], .3)
    return smooth_hanning((sig - th)/th, 20)


# %%
all_sigs = []
for sess in Sessions.fetch(as_dict=True):
    data, rois = get_session_data(sess['mouse'], sess['date'], roi_data_type='raw')
    if data is None: continue
    rois[data.is_rec ==0] = np.nan

    # Get active rois
    _, rois_zscore = get_session_data(sess['mouse'], sess['date'], roi_data_type='zscore')
    zscores = rois_zscore.values[data.is_rec == 1, :]
    maxz = np.max(np.abs(zscores), 0)
    rois_to_keep = [r for r,z in zip(rois.columns, maxz) if z > 2]
    # print(f'Excluded: {len(rois.columns) - len(rois_to_keep)} ROIs')
    rois = rois[rois_to_keep]

    
    tags = get_session_tags(sess['mouse'], sess['date'], 
                        etypes=('visual', 'audio', 'audio_visual'), 
                        ttypes=(TAG))
    
    control_tags = get_session_tags(sess['mouse'], sess['date'], 
                        etypes=('control',), 
                        ttypes=(TAG))

    homing_tags = get_session_tags(sess['mouse'], sess['date'], 
                            etypes=('homing',), 
                            ttypes=(TAG))

    # plot each ROIs signal
    sigs = [[] for i in range(len(rois.columns))]
    for n, roi in track(enumerate(rois.columns), total=(len(rois.columns))):
        roi_mean = rois[roi].mean()

        # make figure
        f, axarr = plt.subplots(ncols=2, nrows=4, figsize=(16, 9), gridspec_kw={'width_ratios': [3, 1]})
        f.suptitle=f"{sess['mouse']} {sess['date']} {roi}"
        ax0 = axarr[0, 0].twinx()  # instantiate a second axes that shares the same x-axis

        # loop over stim evoked    
        for i, tag in tags.reset_index().iterrows():
            # get events frames
            start, end, stim = get_frames_numbers(tag.session_frame, tag.session_stim_frame)
            if data.is_rec[start] == 0: continue

            # plot tracking data
            axarr[0, 0].plot(data.x[start:end].values, lw=2, color=shelt_dist_color, alpha=1)
            ax0.plot(data.s[start:end].values, lw=2, color=speed_color, alpha=.5)

            # plot ROI signal
            sig = dff(rois[roi][start:end].values)
            pre, post = get_pre_vs_post(sig, roi_mean)
            axarr[1, 1].scatter([0, 1], [pre, post], lw=1, edgecolors=[.2, .2, .2], color=signal_color, s=100)
            axarr[1, 1].plot([0, 1], [pre, post], lw=1, color=[.4, .4, .4], zorder=-1)
            axarr[1, 0].plot(sig, lw=2, color=signal_color, alpha=1)

            sigs[n].append(sig)

        # Loop over other tags
        for ax, ax2, col, tgs in zip(axarr[2:, 0], axarr[2:, 1], (cyan, pink), (control_tags, homing_tags)):
            for i, tag in tgs.reset_index().iterrows():
                start, end, stim = get_frames_numbers(tag.session_frame, tag.session_stim_frame)
                if data.is_rec[start] == 0: continue

                sig = rois[roi][start:end].values
                pre, post = get_pre_vs_post(sig, roi_mean)
                ax2.scatter([0, 1], [pre, post], lw=1, edgecolors=[.2, .2, .2], color=col, s=100)
                ax2.plot([0, 1], [pre, post], lw=1, color=[.4, .4, .4], zorder=-1)
                ax.plot(sig, lw=2, color=col, alpha=1)


        # Set axes
        ax0.set(title='Behaviour', ylim=[0, 150], ylabel='speed\n$\\frac{cm}{s}$')
        axarr[0, 0].set(ylim=[0, 80], ylabel='X position\n$cm$', xticks=[])
        axarr[1, 0].set(title='Stim evoked', ylabel=f'{roi}\nDFF', xticks=[])
        axarr[2, 0].set(title='Control tags', ylabel=f'{roi}\nDFF', xticks=[])
        axarr[3, 0].set(title='Homing tags', ylabel=f'{roi}\nDFF', **xlbl)

        axarr[0, 1].axis('off')

        if control_tags.empty:
            axarr[2, 1].axis('off')
        else:
            mark_pre_post(axarr[2, 0])

        if homing_tags.empty:
            axarr[3, 1].axis('off')
        else:
            mark_pre_post(axarr[3, 0])

        mark_pre_post(axarr[1, 0])
        mark_pre_post(axarr[0, 0])

        for ax in axarr[1:, 0]:
            ax.axhline(0, ls='--', lw=1, color=[.3, .3, .3])

        for ax in axarr[1:, 1]:
            # ax.set(xlabel='event number', ylabel='$\\frac{\\mu_{post}-\\mu_{pre}}{\\mu_tot}$')
            ax.set(xticks=[0, 1], xlim=[-.2, 1.2], xticklabels=['pre', 'post'], ylabel='mean sig')

        for ax in axarr[:, 0]:
            ax.axvline(n_frames_pre, lw=1.5, color=[.2, .2, .2])

        clean_axes(f)
        f.tight_layout()
        save_figure(f, fld/f'{sess["mouse"]}_{sess["date"]}_{roi}_dff_alignedto_{TAG}', verbose=False)


        # plt.close(f)
        # break

    all_sigs.extend(sigs)
    # break 

# 
# Average each ROI
averages = []
for sig in all_sigs:
    averages.append(np.mean(np.vstack(sig), 0))
averages = np.array(averages)


'''
    Plt as heatmap
'''
f, ax = plt.subplots(figsize=(16, 9))

m1 = [np.where(a ==  np.nanmax(a))[0] for a in averages]
at_max = np.array([m[0] if len(m) else 0 for m in m1])

cmap = matplotlib.cm.hot
cmap.set_bad('black',1.)
ax.imshow(averages[np.argsort(at_max)], cmap=cmap, interpolation='none')

ax.axvline(n_frames_pre, lw=2, color='w')
ax.set(ylabel='ROI number', **xlbl)

clean_axes(f)
f.tight_layout()
save_figure(f, fld/f'{sess["mouse"]}_{sess["date"]}_dff_alignedto_{TAG}_heatmap', verbose=False)


f, ax = plt.subplots(figsize=(9, 9))
_ = ax.hist((np.array(at_max)- n_frames_pre)/30, bins=50, color='k')
ax.set(ylabel='count', xlabel='Time to max signal')

clean_axes(f)
f.tight_layout()
save_figure(f, fld/f'{sess["mouse"]}_{sess["date"]}_time_to_max_aligned_to_{TAG}', verbose=False)
# %%
