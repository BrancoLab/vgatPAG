# %%
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from rich.progress import track
import pandas as pd
from myterial import cyan, teal, indigo, orange, salmon, grey_light, grey, grey_dark, blue_grey_darker
from myterial import brown_dark, grey, blue_grey_dark, grey_darker, salmon_darker, orange_darker, blue_grey
from scipy.stats import zscore
from collections import namedtuple
from pyrnn.analysis.dimensionality import get_n_components_with_pca
from scipy.stats import ttest_ind, linregress
from statsmodels.stats.multitest import multipletests

from pyinspect import install_traceback
install_traceback()

from fcutils.plotting.utils import calc_nrows_ncols, clean_axes, save_figure
from fcutils.maths.utils import rolling_mean, derivative
from fcutils.plotting.plot_distributions import plot_kde

from vgatPAG.database.db_tables import ManualBehaviourTags, Roi, Sessions
from Analysis import get_session_data, get_session_tags, get_tags_sequences, speed_color, shelt_dist_color, get_active_rois

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
def dff(sig):
    th = np.nanpercentile(sig[:n_frames_pre], .3)
    return rolling_mean((sig - th)/th, 3)

def get_slope(y):
    x = np.arange(len(y))
    lr = linregress(x, y)
    return lr.slope
    
def get_intercept(y):
    x = np.arange(len(y))
    lr = linregress(x, y)
    return lr.intercept

def plot_with_slope(ax, x0, x1, slope, intercept, **kwargs):
    a, b = 0, x1-x0
    y0, y1 = (a * slope) + intercept, (b * slope) + intercept
    ax.plot([x0, x1], [y0, y1], **kwargs)

def get_significant_slopes(slopes):
    ps = [ttest_ind(slopes['baseline'], v).pvalue for v in slopes.values()]
    corrected = multipletests(ps, method='bonferroni')
    return corrected[0]

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
    T = ('STIM', 'H', 'B', 'C', 'E', 'after')
    return start, end , X, Y, T

# %%
lbls = ('baseline', 'stim', 'start', 'run', 'shelter', 'stop')
ACTIVE = dict(mouse=[], date=[], roi=[], active=[])
for sess in Sessions.fetch(as_dict=True):
    print(sess['mouse'], sess['date'])
    data, rois = get_session_data(sess['mouse'], sess['date'], roi_data_type='raw')
    if data is None: continue
    rois[data.is_rec==0] = np.nan

    tags = get_session_tags(sess['mouse'], sess['date'], 
                        etypes=('visual', 'audio', 'audio_visual'), 
                        ttypes=('H', 'B', 'C', 'E'))

    
    # get tags sequences
    sequences = get_tags_sequences(tags)
    if len(set([s.STIM for s in sequences])) <3: continue

    # Loop over ROIs
    for roi in rois.columns:
        slopes = {k:[] for k in ('baseline', 'STIM', 'H', 'B', 'C', 'E', 'after')}
        slopes_colors = {k:c for k,c in zip(slopes.keys(), (grey, salmon, orange, teal, indigo, blue_grey_dark))} 


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
            if seq.E is None or seq.C is None or seq.B is None or seq.H is None: continue
            start, end, X, Y, T = complete_X_Y(seq)
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

            # Get a distribution of signal slopes in baseline
            K = 20
            baseline = sig[n_frames_pre-K:n_frames_pre]
            window = 5
            slope = pd.Series(baseline).rolling(window, min_periods=1, center=True).apply(get_slope, raw=True)
            intercept = pd.Series(baseline).rolling(window, min_periods=1, center=True).apply(get_intercept, raw=True)
            slopes['baseline'].extend(list(slope))
            for n in range(int(len(baseline)/window)):
                x0, x1 = n * window - (window/2) + n_frames_pre-K, (n+1) * window - (window/2) + n_frames_pre-K
                try:
                    plot_with_slope(axarr[1, 0], x0, x1, slope[n*window], intercept[n*window], lw=2, color='g', zorder=100)
                except:
                    pass


            # Plot sequence chunks
            for n, (x, (t0, t1), tag, color) in enumerate(zip(X, Y, T, colors)):
                axarr[1, 0].plot(x, sig[t0:t1], lw=3, color=color, label=lab[n] if not has_legend  else None)
                
                # Get slopes
                if tag != 'after':
                    slope = pd.Series(sig[t1:t1+K]).rolling(window, min_periods=1, center=True).apply(get_slope, raw=True)
                    intercept = pd.Series(sig[t1:t1+K]).rolling(window, min_periods=1, center=True).apply(get_intercept, raw=True)

                    for m in range(int(K/window)):
                        x0, x1 = m * window - (window/3) + 31 , (m+1) * window - (window/3) + 31 
                        try:
                            plot_with_slope(tag_axes[n], x0, x1, slope[m*window], intercept[m*window], lw=3, color=blue_grey_darker, zorder=100)
                        except:
                            pass
                    slopes[tag].extend(list(slope))


                if n > 0 and n < 4:
                    try:
                        axarr[0, 0].scatter(x[0], data.x[t0+start], color=color, lw=1, 
                            edgecolors=[.2, .2, .2], s=100, zorder=100)
                        axarr[1, 0].scatter(x[0], sig[t0], color=color, lw=1, 
                            edgecolors=[.2, .2, .2], s=100, zorder=100,)
                    except:
                        pass
                
                # Plot tags aligned
                try:
                    if n < len(colors) - 1:
                        tag_axes[n].plot(np.arange(30), sig[t1-30:t1], color=colors[n], lw=3, alpha=.6)
                        tag_axes[n].plot(np.arange(30, 60), sig[t1:t1+30], color=colors[n+1], lw=3, alpha=.6)
                        tag_axes[n].scatter(30, sig[t1], s=100, zorder=100, color=colors[n+1], lw=1, edgecolors=[.3, .3, .3])
                except:
                    continue
            has_legend = True

        # Get if ROI active
        significant = get_significant_slopes(slopes)
        ACTIVE['mouse'].append(sess['mouse'])
        ACTIVE['date'].append(sess['date'])
        ACTIVE['roi'].append(roi)
        if np.any(significant):
            ACTIVE['active'].append(True)
        else:
            ACTIVE['active'].append(False)

        # Plot slopes KDE
        for n, (col, (k, slps)) in  enumerate(zip(slopes_colors.values(), slopes.items())):
            bins = np.linspace(-.2, .2, 10)
            if significant[n]:
                alpha = 1
            else:
                alpha = .2
            plot_kde(ax=axarr[1, 3], data=slps, color=col, alpha=alpha, normto=.8, z=6-n, label=k)
        axarr[1, 3].axvline(0, color=[.5, .5, .5], lw=2, zorder=-1)
        axarr[1, 3].set(yticks=[], xlabel='Mean slope', ylabel='Slopes distribution')


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


        axes, titles = tag_axes, ('stim', 'start', 'run', 'shelter', 'stop')
        for ax, ttl in zip(tag_axes, titles):
            ax.set(title=ttl)
            ax.set(xticks=[0, 30, 60], xticklabels=[-1, 0, 1], xlabel='s', yticks=[], ylim=axarr[1, 0].get_ylim())

            ax.spines['left'].set_visible(False)
            ax.axvline(30, color=[.2, .2, .2], lw=3)

        save_figure(f, fld/f'{sess["mouse"]}_{sess["date"]}__{roi}_active_{ACTIVE["active"][-1]}', verbose=False)
        # plt.close(f)
        break
    break
pd.DataFrame(ACTIVE).to_hdf('ACTIVE_ROIS.h5', key='hdf')

# %%
'''
Make a similar plot but looking at the first PC of each FOV
'''
for sess in Sessions.fetch(as_dict=True):
    print(sess['mouse'], sess['date'])
    data, rois = get_session_data(sess['mouse'], sess['date'], roi_data_type='raw')
    if data is None: continue

    tags = get_session_tags(sess['mouse'], sess['date'], 
                        etypes=('visual', 'audio', 'audio_visual'), 
                        ttypes=('H', 'B', 'C', 'E'))
    
    # get tags sequences
    sequences = get_tags_sequences(tags)
    if len(set([s.STIM for s in sequences])) <3: continue

    # get active ROIs
    rois = get_active_rois(rois, sequences, sess)

    signals = rois.values[data.is_rec==1, :].astype(np.float32)
    scaler = StandardScaler().fit(signals)

    pca = PCA(n_components=1).fit(scaler.transform(signals))

    pc = pca.transform(scaler.transform(rois.values))


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
        if seq.E is None or seq.C is None or seq.B is None or seq.H is None: continue
        start, end, X, Y, T = complete_X_Y(seq)
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
        sig = pc[start:end]

        # Plot sequence chunks
        for n, (x, (t0, t1), tag, color) in enumerate(zip(X, Y, T, colors)):
            axarr[1, 0].plot(x, sig[t0:t1], lw=3, color=color, label=lab[n] if not has_legend  else None)
            
            if n > 0 and n < 4:
                try:
                    axarr[0, 0].scatter(x[0], data.x[t0+start], color=color, lw=1, 
                        edgecolors=[.2, .2, .2], s=100, zorder=100)
                    axarr[1, 0].scatter(x[0], sig[t0], color=color, lw=1, 
                        edgecolors=[.2, .2, .2], s=100, zorder=100,)
                except:
                    pass
            
            # Plot tags aligned
            try:
                if n < len(colors) - 1:
                    tag_axes[n].plot(np.arange(30), sig[t1-30:t1], color=colors[n], lw=3, alpha=.6)
                    tag_axes[n].plot(np.arange(30, 60), sig[t1:t1+30], color=colors[n+1], lw=3, alpha=.6)
                    tag_axes[n].scatter(30, sig[t1], s=100, zorder=100, color=colors[n+1], lw=1, edgecolors=[.3, .3, .3])
            except:
                continue
        has_legend = True

    # style axes
    axarr[1, 0].legend()
    ax0.set(title='Behaviour', ylim=[0, 150], ylabel='speed\n$\\frac{cm}{s}$')

    if longest == 0: continue
    x = np.arange(60, longest+30, 30)
    axarr[1, 0].set(title='Signal', ylabel=f'FIRST PC ON ACTIVE ROIS', 
                xlim=[x.min(), x.max()],
                xticks=x, xticklabels=((x-n_frames_pre)/30).astype(np.int32), xlabel='s')
    axarr[0, 0].set(ylim=[0, 80], ylabel='X position\n$cm$', xticks=[], xlim=[x.min(), x.max()],)
    clean_axes(f)
    axarr[-1, -1].axis('off')

    axes, titles = tag_axes, ('stim', 'start', 'run', 'shelter', 'stop')
    for ax, ttl in zip(tag_axes, titles):
        ax.set(title=ttl)
        ax.set(xticks=[0, 30, 60], xticklabels=[-1, 0, 1], xlabel='s', yticks=[], ylim=axarr[1, 0].get_ylim())

        ax.spines['left'].set_visible(False)
        ax.axvline(30, color=[.2, .2, .2], lw=3)

    save_figure(f, fld/f'{sess["mouse"]}_{sess["date"]}__FIRST_PC', verbose=False)
    plt.close(f)

    
    # break


# %%

# %%

# %%
