# %%
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

from pyinspect import install_traceback
install_traceback()

from fcutils.plotting.utils import calc_nrows_ncols, clean_axes, save_figure
from fcutils.maths.utils import rolling_mean, derivative
from vgatPAG.database.db_tables import ManualBehaviourTags, Roi, Sessions
from Analysis import get_session_data, signal_color, get_session_tags, speed_color, shelt_dist_color

# %%

pre_pos_s = 1.5
pre_pos = int(pre_pos_s*30)
n_s_pre, n_s_post = 5, 2

n_frames_pre = n_s_pre * 30
n_frames_post = n_s_post * 30
xlbl = dict(
        xlabel='time from tag\n(s)', 
        xlim=[0, n_frames_post + n_frames_pre],
        xticks=[0, n_frames_pre - pre_pos, n_frames_pre, n_frames_pre+pre_pos, n_frames_pre+n_frames_post], 
        xticklabels=[-n_s_pre, -pre_pos_s, 0, pre_pos_s, n_s_post]
        )


fld = Path('D:\\Dropbox (UCL)\\Project_vgatPAG\\analysis\\doric\\Fede\\\ddf_tag_aligned_all_tags_NOSMOOTHING')

# %%
def get_frames_numbers(event_frame, stim_frame):
        start, end = event_frame-n_frames_pre, event_frame+n_frames_post
        stim = n_frames_pre - (event_frame - stim_frame)
        if stim == n_frames_pre:
            stim = None
        return start, end, stim

def dff(sig):
    th = np.nanpercentile(sig[:n_frames_pre], .3)
    return rolling_mean((sig - th)/th, 3)
    # return np.array((sig - th)/th)

def get_tags_sequences(tags):
    sequence = namedtuple('sequence', 'STIM, H, B, C, E, D')
    sequences = []
    
    # get stimuli
    stims = tags.session_stim_frame.unique()

    for n, stim in enumerate(stims):
        seq = [stim]
        tgs = tags.loc[(tags.session_stim_frame == stim)&(tags.session_frame>=stim)]
        if n < len(stims)-1:
            tgs = tgs.loc[tgs.session_frame < stims[n+1]]

        for ttype in ('H','B','C','E', 'D'):
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

def rel(event, stim):
    return event - stim + n_frames_pre

def complete_X_Y(seq):
    start = seq.STIM - n_frames_pre
    end = seq.E + n_frames_post

    # define x/y coord sets for plotting
    stim = rel(seq.STIM, seq.STIM)
    H = rel(seq.H, seq.STIM)
    B = rel(seq.B, seq.STIM)
    C = rel(seq.C, seq.STIM)
    E = rel(seq.E, seq.STIM)

    # xonset = np.arange(onset)
    xstim = np.arange(0, stim)
    xH = np.arange(stim, H)
    xB = np.arange(H, B)
    xC = np.arange(B, C)
    xE = np.arange(C, E)
    xend = np.arange(E, E+n_frames_post)

    X = (xstim, xH, xB, xC, xE, xend)
    Y = ((0, stim),  (stim, H), (H, B), (B, C), (C, E), (E, rel(end, seq.STIM)))

    return start, end , X, Y

def incomplete_X_Y(seq):
    start = seq.STIM - n_frames_pre
    end = seq.D + n_frames_post

    stim = rel(seq.STIM, seq.STIM)
    H = rel(seq.H, seq.STIM)
    B = rel(seq.B, seq.STIM) if seq.B is not None else None
    D = rel(seq.D, seq.STIM)

    xstim = np.arange(0, stim)
    xH = np.arange(stim, H)
    if B is not None:
        xB = np.arange(H, B)
        xD = np.arange(B, D)
    else:
        xD = np.arange(H, D)
    xend = np.arange(D, D+n_frames_post)

    if B is not None:
        X = (xstim, xH, xB, xD, xend)
        Y = ((0, stim),  (stim, H), (H,B), (B,D), (D, rel(end, seq.STIM)))
        colors = (grey, salmon_darker, orange_darker, blue_grey_dark, grey)
    else:
        X = (xstim, xH, xD, xend)
        Y = ((0, stim),  (stim, H), (H,D), (D, rel(end, seq.STIM)))
        colors = (grey, salmon_darker, blue_grey_dark, grey)

    return start, end , X, Y, colors


def get_onset():
    th = np.nanmean(sig[:n_frames_pre]) + 1.5*np.nanstd(sig[:n_frames_pre])
    try:
        above = np.where(sig >= th)[0][0] - 1
    except:
        onset = rel(seq.STIM, seq.STIM)
    else:
        try:
            onset = np.where(np.diff(sig[:above])<=0)[0][-1]+1
        except:
            onset = rel(seq.STIM, seq.STIM)

# %%
all_sigs = []
lbls = ('baseline', 'stim', 'start', 'run', 'shelter', 'stop')
for sess in Sessions.fetch(as_dict=True):
    print(sess['mouse'], sess['date'])
    data, rois = get_session_data(sess['mouse'], sess['date'], roi_data_type='raw')
    if data is None: continue
    rois[data.is_rec==0] = np.nan

    tags = get_session_tags(sess['mouse'], sess['date'], 
                        etypes=('visual', 'audio', 'audio_visual'), 
                        ttypes=('H', 'B', 'C', 'E', 'D'))

    
    # get tags sequences
    sequences = get_tags_sequences(tags)
    if len(set([s.STIM for s in sequences])) <3: continue

    # Loop over ROIs
    for roi in rois.columns:
        # Create a figure
        f, axarr = plt.subplots(nrows=2, ncols=4, figsize=(18, 9), sharex=False,  gridspec_kw={'width_ratios': [4, 1, 1, 1]})

        ax0 = axarr[0, 0].twinx()
        axarr[0, 0].axvline(n_frames_pre, lw=2, color=salmon)
        axarr[1, 0].axvline(n_frames_pre, lw=2, color=salmon)
        tag_axes = [axarr[0, 1], axarr[0, 2], axarr[0, 3], axarr[1, 1], axarr[1, 2], ]
        incomplete_tag_axes = [axarr[0, 1], axarr[0, 2], axarr[1, 3]]
        incomplete_tag_axes2 = [axarr[0, 1], axarr[0, 2], axarr[0, 3], axarr[1, 3]]

        # Loop over sequences
        traces = []
        prev_stim, has_legend, longest = 0, False, 0
        for sn, seq in enumerate(sequences):
            # check we didn't have a stim too  close
            if seq.STIM - prev_stim < 60*30:
                print('Stim too close')
                continue
            else:
                pre_stim = seq.STIM

            # Check for aborted escapes
            if seq.D is not None:
                if seq.H is None: 
                    continue  # ignored stim
                if seq.E is not None: raise ValueError

                start, end, X, Y, colors = incomplete_X_Y(seq)
                x_col, s_col = brown_dark, grey
                if seq.B is None:
                    lab = ('baseline', 'stim', 'start', 'abort')
                else:
                    lab = ('baseline', 'stim', 'start', 'run', 'abort')

            else:
                if seq.E is None or seq.C is None or seq.B is None or seq.H is None: continue
                start, end, X, Y = complete_X_Y(seq)
                colors = (grey, salmon, orange, teal, indigo, grey)
                x_col, s_col = shelt_dist_color, speed_color
                lab = lbls
            if data.is_rec[start] == 0: continue  # stim when not recording
            if end-start > longest:
                longest = end - start

            # plot tracking
            axarr[0, 0].plot(data.x[start:end].values, lw=2, color=x_col, alpha=1)
            ax0.plot(data.s[start:end].values, lw=2, color=s_col, alpha=.5)

            # plot ROI signal
            sig = dff(rois[roi][start:end])
            trace = np.full((600), np.nan)
            trace[:len(sig)] = sig
            traces.append(trace)

            # Plot sequence chunks
            for n, (x, (t0, t1), color) in enumerate(zip(X, Y, colors)):
                axarr[1, 0].plot(x, sig[t0:t1], lw=3, color=color, label=lab[n] if not has_legend  else None)
                
                M = 4 if seq.B is None else 4
                L = len(colors) if seq.D is None else M
                if n > 0 and n < L:
                    try:
                        axarr[0, 0].scatter(x[0], data.x[t0+start], color=color, lw=1, 
                            edgecolors=[.2, .2, .2], s=100, zorder=100)
                        axarr[1, 0].scatter(x[0], sig[t0], color=color, lw=1, 
                            edgecolors=[.2, .2, .2], s=100, zorder=100,)
                    except:
                        pass
                
                # Plot tags aligned
                if seq.D is None:
                    try:
                        if n < len(colors) - 1:
                            tag_axes[n].plot(np.arange(30), sig[t1-30:t1], color=colors[n], lw=3)
                            tag_axes[n].plot(np.arange(30, 60), sig[t1:t1+30], color=colors[n+1], lw=3)
                            tag_axes[n].scatter(30, sig[t1], s=100, zorder=100, color=colors[n+1], lw=1, edgecolors=[.3, .3, .3])
                    except:
                        continue
                else:
                    if seq.B is None:
                        if n < 3:
                            incomplete_tag_axes[n].plot(np.arange(30), sig[t1-30:t1], color=colors[n], lw=3)
                            incomplete_tag_axes[n].plot(np.arange(30, 60), sig[t1:t1+30], color=colors[n+1], lw=3)
                            incomplete_tag_axes[n].scatter(30, sig[t1], s=100, zorder=100, color=colors[n+1], lw=1, edgecolors=[.3, .3, .3])
                    else:
                        if n < 4:
                            incomplete_tag_axes2[n].plot(np.arange(30), sig[t1-30:t1], color=colors[n], lw=3)
                            incomplete_tag_axes2[n].plot(np.arange(30, 60), sig[t1:t1+30], color=colors[n+1], lw=3)
                            incomplete_tag_axes2[n].scatter(30, sig[t1], s=100, zorder=100, color=colors[n+1], lw=1, edgecolors=[.3, .3, .3])
            has_legend = True

        # plot mean signal
        # mn = np.median(np.vstack(traces), 0)
        # axarr[1].plot(mn, lw=4, color='r', zorder=-2)
        # axarr[1].plot(mn, lw=2, color='k', zorder=200, label='MEDIAN')

        # style axes
        axarr[1, 0].legend()
        ax0.set(title='Behaviour', ylim=[0, 150], ylabel='speed\n$\\frac{cm}{s}$')

        if longest == 0: continue
        x = np.arange(60, longest+30, 30)
        axarr[1, 0].set(title='Signal', ylabel=f'{roi}\nDFF', 
                    xlim=[x.min(), x.max()],
                    xticks=x, xticklabels=((x-n_frames_pre)/30).astype(np.int32), xlabel='s')
        axarr[0, 0].set(ylim=[0, 80], ylabel='X position\n$cm$', xticks=[], xlim=[x.min(), x.max()],)
        clean_axes(f)


        axes, titles = tag_axes + [axarr[1, 3]], ('stim', 'start', 'run', 'shelter', 'stop', 'abort')
        for ax, ttl in zip(axes, titles):
            ax.set(title=ttl)
            ax.set(xticks=[0, 30, 60], xticklabels=[-1, 0, 1], xlabel='s', yticks=[], ylim=axarr[1, 0].get_ylim())
            ax.spines['left'].set_visible(False)
            ax.axvline(30, color=[.2, .2, .2], lw=3)

        # save_figure(f, fld/f'{sess["mouse"]}_{sess["date"]}__{roi}', verbose=False)
        # plt.close(f)
        break
    break
# %%
'''
    Tag aligned PCA of population activity
'''

for sess in Sessions.fetch(as_dict=True):
    print(sess['mouse'], sess['date'])
    data, rois = get_session_data(sess['mouse'], sess['date'], roi_data_type='raw')
    if data is None: continue
    rois[data.is_rec==0] = np.nan

    tags = get_session_tags(sess['mouse'], sess['date'], 
                        etypes=('visual', 'audio', 'audio_visual'), 
                        ttypes=('H', 'B', 'C', 'E', 'D'))

    
    # get tags sequences
    sequences = get_tags_sequences(tags)
    if len(set([s.STIM for s in sequences])) <3: continue

    f, axarr = plt.subplots(nrows=5, figsize=(14, 18), sharex=True)
    prev_stim = 0
    Rs = []
    for sn, seq in enumerate(sequences):
        # check we didn't have a stim too  close
        if seq.STIM - prev_stim < 60*30:
            continue
        else:
            pre_stim = seq.STIM

        # Check for aborted escapes
        if seq.D is not None:
            continue
        else:
            if seq.E is None or seq.C is None or seq.B is None or seq.H is None: continue
            start, end, X, Y = complete_X_Y(seq)
            colors = (grey, salmon, orange, teal, indigo, grey)
        if data.is_rec[start] == 0: continue  # stim when not recording

        # Compute DFF
        dff_rois= rois[start:end].apply(dff)

        R = StandardScaler().fit_transform(dff_rois.values.astype(np.float32))
        Rs.append(R)
        pca = PCA(n_components=5).fit(R)

        pcs = pca.transform(R)
        for  ax,pc in zip(axarr, pcs.T):
            for n, (x, (t0, t1), color) in enumerate(zip(X, Y, colors)):
                if color == grey:
                    zorder, lw = 1, 2
                else:
                    zorder, lw = 2, 3
                ax.plot(x, pc[t0:t1],   color=color, lw=lw, zorder=zorder)
                if n > 0:
                    ax.scatter(t0, pc[t0], lw=2, s=50,  edgecolors=[.2, .2, .2], zorder=100, color=color)
            ax.axvline(n_frames_pre, lw=2, color=salmon, zorder=-1)

    for n,ax in enumerate(axarr):
        ax.set(ylabel=f'PC {n}')
    x = np.arange(60, 600, 30)
    axarr[-1].set(
                xlim=[x.min(), x.max()],
                xticks=x, xticklabels=((x-n_frames_pre)/30).astype(np.int32), xlabel='s')
    clean_axes(f)
    save_figure(f, fld/f'{sess["mouse"]}_{sess["date"]}__PCA', verbose=False)
    plt.close(f)
    # Get the number of components needed to explain 85%  of variance
    # if Rs:
    #     get_n_components_with_pca(np.vstack(Rs))
    # break
# %%
# %%


# %%
