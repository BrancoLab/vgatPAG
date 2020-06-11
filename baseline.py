"""
    Look at the activity of each ROI aligned to stimuli onsets
    alongside behavioural variables like speed etc.
"""
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
from random import choices

from Analysis  import (
        mice,
        sessions,
        recordings,
        recording_names,
        stimuli,
        clean_stimuli,
        get_mouse_session_data,
        sessions_fps,
)

from Analysis import (
    shelt_dist_color,
    speed_color,
    ang_vel_colr,
    signal_color,
)

from fcutils.plotting.utils import save_figure, clean_axes, set_figure_subplots_aspect
from fcutils.plotting.plot_elements import plot_mean_and_error, plot_shaded_withline
from fcutils.plotting.plot_distributions import plot_kde
from fcutils.maths.filtering import line_smoother

from brainrender.colors import makePalette, colorMap

from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import statsmodels.api as sm
from sklearn import preprocessing


from vgatPAG.paths import output_fld
output_fld = Path(output_fld)



# %%
'''
    Get baselines
'''
# Variables
N_sec_pre = 2
N_sec_post = 8
SHOW_MEAN = False

STIMS = clean_stimuli

exploration_baselines = {} #  each ROIs baseline before the first stimulus
exploration_baselines_behaviour = {}
baselines = {}
behaviour_baselines = {}
for mouse in mice:
    for sess in tqdm(sessions[mouse]):
        frames_pre = N_sec_pre * sessions_fps[mouse][sess]

        # Prep data
        stims = STIMS[f'{mouse}-{sess}']
        if not len(stims.all): 
            baselines[f'{mouse}-{sess}'] = None
            continue
        tracking, ang_vel, speed, shelter_distance, signals, nrois, is_rec = get_mouse_session_data(mouse, sess, sessions)
        
        bss = [[] for i in np.arange(nrois)] # signal baselines
        xss = [] # x position
        sss = [] # speed
        ess = [] # exploration baselines

        for stim in stims.all:
            if is_rec[stim] == 0: continue # keep only stims that happened during recordings 
            for n, signal in enumerate(signals):
                bss[n].append(signal[stim-frames_pre:stim])

            xss.append(tracking.x[stim-frames_pre:stim])
            sss.append(speed[stim-frames_pre:stim])

        # Get exploration baseline
        for n, signal in enumerate(signals):
            sig = signal[:stims.all[0]]
            ess.append(sig[is_rec[:stims.all[0]]==1])

        exploration_baselines_behaviour[f'{mouse}-{sess}'] = (tracking.x[:stims.all[0]][is_rec[:stims.all[0]]==1], 
                                                            speed[:stims.all[0]][is_rec[:stims.all[0]]==1])


        exploration_baselines[f'{mouse}-{sess}'] = np.vstack(ess) # N rois by N frames exploration
        baselines[f'{mouse}-{sess}'] = [np.vstack(bs) for bs in bss] # each rois bs array is N stim by N frames pre
        behaviour_baselines[f'{mouse}-{sess}'] = (np.vstack(xss), np.vstack(sss)) # N stims by N frames


# %%
"""
    Fit PCA to the whole sessions recording
"""
whole_sess_pca = {}
for mouse in mice:
    for sess in sessions[mouse]:
        # Prep data
        tracking, ang_vel, speed, shelter_distance, signals, nrois, is_rec = get_mouse_session_data(mouse, sess, sessions)

        signal = np.vstack([s[is_rec == 1] for s in signals]) # n rois by n frames
        whole_sess_pca[f'{mouse}-{sess}'] = PCA(n_components=2).fit(signal.T)

        # plt.figure()
        # plt.plot(whole_sess_pca[f'{mouse}-{sess}'].explained_variance_ratio_)




# %%

"""
    Plot for each stim position speed etc, + 
    signal trace for each ROI +
    avg + std during baseline

"""

# Variables
N_sec_pre = 5
N_sec_post = 8
SHOW_MEAN = False

for mouse in mice:
    for sess in sessions[mouse]:
        # Prep data
        stims = stimuli[f'{mouse}-{sess}']
        tracking, ang_vel, speed, shelter_distance, signals, nrois, is_rec = get_mouse_session_data(mouse, sess, sessions)

        frames_pre = N_sec_pre * sessions_fps[mouse][sess]
        frames_post = N_sec_post * sessions_fps[mouse][sess]
        time = np.arange(-frames_pre, frames_post)

        raise NotImplementedError('fix baselines usage')
        baselines = [[] for i in np.arange(nrois)]

        # Plot traces
        f, axarr = plt.subplots(nrows=3 + nrois, figsize=(12, 24), sharex=True)
        for stim in tqdm(stims.all):
            try:
                at_shelter = np.where(shelter_distance[stim:stim+frames_post] <= 0)[0][0] + stim + sessions_fps[mouse][sess]
                _time = np.arange(-frames_pre, at_shelter-stim)
            except:
                continue
                _time = time
                at_shelter = stim+frames_post

            axarr[0].plot(_time, shelter_distance[stim-frames_pre:at_shelter], c=shelt_dist_color,
                        alpha=.8)
            axarr[1].plot(_time, speed[stim-frames_pre:at_shelter], c=speed_color,
                        alpha=.8)
            axarr[2].plot(_time, ang_vel[stim-frames_pre:at_shelter], c=ang_vel_colr,
                        alpha=.8)
            
            for n, ax in enumerate(axarr[3:]):
                ax.plot(_time, signals[n][stim-frames_pre:at_shelter], c=signal_color, alpha=.3)
                baselines[n].append(signals[n][stim-frames_pre:stim])

        # Plt baselines
        if SHOW_MEAN:
            for n, baseline in enumerate(baselines):
                mean = np.nanmean(np.vstack(baseline), axis=0)
                std = np.std(np.vstack(baseline), axis=0)

                plot_mean_and_error(mean, std, axarr[3+n], x=time, color=[.2, .2, .2], err_color=[.4, .4, .4])


        # Clean up figure
        ttls = ['Shelter distance', 'speed', 'angular velocity']
        for n, ax in enumerate(axarr):
            ax.axvline(0, lw=2, ls='--', color=[.4, .4, .4])

            if n < len(ttls):
                ttl = ttls[n]
            else:
                ttl = 'signal'
            ax.set(ylabel=ttl)

        axarr[-1].set(xlabel='frames')
        clean_axes(f)
        break
    break

# %%
"""
        Plot each trials baseline as a scatter plot + scatter of PCA
"""


def fit_lin_regrs(R, X, S):
    """
        predicts a ROI's mean baseline activity at each trial (R)
        with the mean X position (X) and mean speed (S) at each trial
        with a linear regression + OLS
    """
    # prep data
    exog = pd.DataFrame(dict(S=S)).interpolate()
    endog = pd.DataFrame(dict(R=R))

    # NOrmalize and prepare endog
    # exog = pd.DataFrame(preprocessing.scale(exog.values, axis=0), columns=exog.columns, index=exog.index)
    exog = sm.add_constant(exog, prepend=False)

    # Fit and save
    model = sm.OLS(endog, exog).fit()
    return exog, endog, model
    

def get_exploration_mean_and_variance(expl, N, M=1):
    """
        Given a 1d array with a ROIs exploration trace (expl) it takes
        the mean and starndard deviation of N intervals
    """

    idxs = np.arange(frames_pre+1, len(expl)-frames_pre-1)
    starts = choices(idxs, k=N)

    intervals = [expl[i-frames_pre:i] for i in starts]
    data = np.vstack(intervals)

    mean, std = np.mean(data), np.std(data)

    return np.array([mean for i in np.arange(M)]), np.array([std for i in np.arange(M)])




save_fld = output_fld / 'baseline'
save_fld.mkdir(exist_ok=True)

baseline_averages = {}

for mouse in mice:
    for sess in sessions[mouse]:
        frames_pre = N_sec_pre * sessions_fps[mouse][sess]

        # Prep data, make figure
        bss = baselines[f'{mouse}-{sess}']
        if bss is None: continue

        n_trials = bss[0].shape[0]
        n_rois = len(bss)

        f, axarr = plt.subplots(nrows=6, ncols=4, sharex=True, sharey=False, figsize=(24, 10), num=f'{mouse}-{sess}')
        f.suptitle(f'Baseline variability - {mouse} {sess}')
        axarr = axarr.flatten()
        
        # Plot each trial for each ROI
        colors = np.array([colorMap(i, 'Reds', vmin=-4, vmax=n_trials+4) for i in np.arange(n_trials)])

        # Get means for speed and X baselines
        x, spd = behaviour_baselines[f'{mouse}-{sess}']
        x, spd = np.nanmean(x, 1), np.nanmean(spd, 1)


        rois_means = [] #  used for PCA
        for n, (ax, bs) in enumerate(zip(axarr, bss)):
            # get rois data
            means = np.array([np.mean(b) for b in bs])
            rois_means.append(means)
            
            stds = np.array([np.std(b)*2 for b in bs])
            sortidx = np.argsort(means)[::-1]

            # Fit linear regression model
            exog, endog, model = fit_lin_regrs(means, x, spd)

            # Plot not sorted
            time = np.arange(len(means))
            ax.errorbar(time, means, fmt='o', yerr=stds, capthick=2, ecolor='k', zorder=30)
            ax.scatter(time, means, c=colors, s=50, zorder=99, lw=1, edgecolors ='k')

            # Plot speed and X baselines scaled
            _x, _s = minmax_scale(x, feature_range=(np.min(means), np.max(means))), minmax_scale(spd, feature_range=(np.min(means), np.max(means)))
            # ax.plot(time, _x, color=shelt_dist_color, lw=4, alpha=.2, zorder=20, label='X pos')
            # ax.plot(time, _s, color=speed_color, lw=4, alpha=.6, zorder=20, label='Speed')

            # Plot linear regression's prediction
            # ax.scatter(time, model.fittedvalues, color=[.5, .5, .5], s=45, zorder=90, alpha=.5)

            # Plot exploration ROI mean activity
            expl_mean, expl_sdt = get_exploration_mean_and_variance(exploration_baselines[f'{mouse}-{sess}'][n, :], n_trials*5, n_trials)
            plot_mean_and_error(expl_mean, expl_sdt, ax, x=time, color=[.2, .2, .2], err_color=[.4, .4, .4], label='expl mean')

            # Clean up axis
            if n == 0:
                ax.legend()
            ax.set(ylabel=f'ROI - {n}', xlabel='stimuli') # , ylim=[np.min(means)-50, np.max(means)+50])

        baseline_averages[f'{mouse}-{sess}'] = pd.DataFrame(rois_means) # used for PCA


        clean_axes(f)

        plt.figure(f'{mouse}-{sess}')
        set_figure_subplots_aspect(hspace=.6, top=.95, wspace=.25)
        save_figure(f, str(save_fld/f'{mouse} {sess} ROIs'))

    #     break
    # break






# %%
"""
    Similar to above but in PCA space
"""
WHOLE_SESS_PCA = 'expl'


for mouse in mice:
    for sess in sessions[mouse]:
        # Prep data, make figure
        bss = baselines[f'{mouse}-{sess}']
        if bss is None: continue

        n_trials = bss[0].shape[0]
        n_rois = len(bss)
        colors = np.array([colorMap(i, 'Reds', vmin=-4, vmax=n_trials+4) for i in np.arange(n_trials)])


        # f2, ax2= plt.subplots(num=f'{mouse}-{sess}2')
        f2 = plt.figure(figsize=(8, 8))
        ax = f2.add_subplot(1, 1, 1)
        f2.suptitle(f'Baseline variability PCA - {mouse} {sess} \n      red = late in session')

        # plot activity in PC space for each trial
        trials = [np.vstack([bss[n][t] for n in np.arange(n_rois)]) for t in np.arange(n_trials)]
        all_trials = np.hstack(trials) # N rois by (N frames * N stimuli)

        # fit PCA on the whole dataset
        if not WHOLE_SESS_PCA:
            pca = PCA(n_components=2).fit(all_trials.T) # Ntrials x (NROIS by nframes)
        elif WHOLE_SESS_PCA == 'expl': # fit PLCA to exploration data
            pca = PCA(n_components=2).fit(np.vstack(exploration_baselines[f'{mouse}-{sess}']).T) 
        else:
            pca = whole_sess_pca[f'{mouse}-{sess}']

        # Plot population activity as PCA
        for t, trial in enumerate(trials):
            PC = pca.transform(np.vstack(trial).T)
 
            ax.errorbar(np.mean(PC[:, 0]), np.mean(PC[:, 1]), xerr=np.std(PC[:, 0]), yerr=np.std(PC[:, 1]), ecolor='k')
            ax.scatter(np.mean(PC[:, 0]), np.mean(PC[:, 1]), color=colors[t], zorder=99, s=60, lw=1, edgecolors ='k')

        # Plot contour of PCA of exploration
        PC = pca.transform(np.vstack(exploration_baselines[f'{mouse}-{sess}']).T)        
        sns.kdeplot(PC[:, 0], PC[:, 1], shade=True, ax=ax, shade_lowest=False, n_levels=10, label='exploration')
        ax.legend()
      

        # Clean up
        ax.set(title='activity PCA', xlabel='PC1', ylabel='PC2')
        clean_axes(f2)
        # set_figure_subplots_aspect(hspace=.6, top=.95, wspace=.25)
        save_figure(f2, str(save_fld/f'{mouse} {sess} PCA'))


    #     break
    # break



# %%
"""
    Look at first PC during exploration and 
    compare to speed trace
"""
for mouse in mice:
    for sess in sessions[mouse]:
        bss = baselines[f'{mouse}-{sess}']
        if bss is None: continue

        pc = PCA(n_components=1).fit_transform(np.vstack(exploration_baselines[f'{mouse}-{sess}']).T) 
        expl_x = exploration_baselines_behaviour[f'{mouse}-{sess}'][0]
        expl_speed = exploration_baselines_behaviour[f'{mouse}-{sess}'][1]

        in_shelt = np.ones_like(expl_x.ravel())
        in_shelt[expl_x.ravel() > 300] = 0


        f, axarr = plt.subplots(nrows=2, sharex=True, figsize=(16, 9))

        data = pd.DataFrame(dict(pc=line_smoother(pc.ravel(), window_size=101), 
                                x=line_smoother(expl_x, window_size=101), 
                                in_shelt = np.cumsum(in_shelt),
                                speed = line_smoother(expl_speed, window_size=101))).interpolate(axis=0)

        plot_shaded_withline(axarr[0], np.arange(len(data)), minmax_scale(data.pc, feature_range=(0, 1)), 
                                            alpha=.3, lw=2, label='pc', color=signal_color)
        # axarr[0].plot(minmax_scale(data.speed, feature_range=(0, 1)), alpha=.4, label='speed')
        axarr[0].plot(minmax_scale(data.x, feature_range=(0, 1)), alpha=.8, label='x', lw=3, zorder=100, color=speed_color)
        
        
        plot_shaded_withline(axarr[1], np.arange(len(data)), data.in_shelt, alpha=.2, 
                        color=shelt_dist_color,  label='in_shelt')
        
        axarr[0].legend()

        axarr[0].set(title=f'{mouse} - {sess} - Pearson corr\n   x-PC {round(data.corr().x.pc, 3)}', ylabel='norm. values')
        axarr[1].set(title='comulative time in shelter',ylabel='# frames', xlabel='frames')

    #     break
    # break



# %%
