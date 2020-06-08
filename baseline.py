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

from Analysis  import (
        mice,
        sessions,
        recordings,
        print_recordings_tree,
        recording_names,
        stimuli,
        get_mouse_session_data,
        sessions_fps,
)

from Analysis import (
    shelt_dist_color,
    speed_color,
    ang_vel_colr,
    signal_color,
)

print_recordings_tree()

from fcutils.plotting.utils import save_figure, clean_axes
from fcutils.plotting.plot_elements import plot_mean_and_error

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

