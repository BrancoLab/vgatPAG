# %%
import numpy as np
import matplotlib.pyplot as plt
from rich.progress import track
from sklearn.preprocessing import minmax_scale, normalize
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from myterial import light_blue_dark, salmon_dark, purple_dark, salmon_darker, light_blue_darker
from scipy.stats import spearmanr, zscore
from pathlib import Path
import pandas as pd
import statsmodels.api as sm

from Analysis  import (
        mice,
        sessions,
        recordings,
        mouse_sessions,
        get_mouse_session_data,
)

from Analysis.misc import get_tiff_starts_ends, get_chunk_rolling_mean_subracted_signal, get_chunked_dff

from fcutils.maths.utils import derivative
from fcutils.plotting.utils import save_figure, clean_axes

fld = Path('D:\\Dropbox (UCL)\\Project_vgatPAG\\analysis\\doric\\Fede\\slow_correlations')

# %%
DO = dict(
    plot_dff_vs_speed=False,
    pearson_corr=True
)

# %%
"""
    Look at correlations between slow changes in ROI activity and slow changes in behaviour
"""
hanning=6
window=100
for mouse, sess, sessname in track(mouse_sessions, description='plotting speed correlation'):
    if not DO['plot_dff_vs_speed']: 
        break
    # Get data
    tracking, ang_vel, speed, shelter_distance, dffs, signals, nrois, is_rec, roi_ids = \
                        get_mouse_session_data(mouse, sess, sessions, hanning_window=hanning)

    # Get chunks start end times
    tiff_starts, tiff_ends = get_tiff_starts_ends(is_rec)

    # Slow behav
    _, speed = get_chunk_rolling_mean_subracted_signal(speed, tiff_starts, tiff_ends, window=window)
    _, shelter_distance = get_chunk_rolling_mean_subracted_signal(shelter_distance, tiff_starts, tiff_ends, window=window)

    speed = minmax_scale(speed * is_rec, feature_range=(-1, 1))
    shelter_distance = minmax_scale(shelter_distance * is_rec, feature_range=(-1, 1))

    rec_off = np.where(is_rec==0)[0][0]
    speed -= speed[rec_off]
    shelter_distance -= shelter_distance[rec_off]

    X = np.arange(len(is_rec))


    # Make folder
    sfld = fld/sessname
    sfld.mkdir(exist_ok=True)

    # loop over rois
    for n, (sig, rid) in enumerate(zip(signals, roi_ids)):
        f, ax = plt.subplots(figsize=(16, 9), sharex=True)


        # get chunked dff
        dff = get_chunked_dff(sig, tiff_starts, tiff_ends)

        diff, smoothed = get_chunk_rolling_mean_subracted_signal(dff, tiff_starts, tiff_ends, window=window)

        smoothed = minmax_scale(smoothed, feature_range=(-1, 1))

        # plot
        # plot_line_outlined(ax, X, shelter_distance, outline=2, 
        #             outline_color=[.3, .3, .3],  lw=1,
        #             color=purple_dark, 
        #             label='shelter distance')
        
        ax.plot(X, speed, lw=3,
                    color=salmon_dark, 
                    label='seed', alpha=.5) 

        ax.plot(X, smoothed - smoothed[rec_off], lw=3,
                    color=light_blue_dark, 
                    label='DFF', alpha=.5) 

                    
        
        ax.legend()
        ax.set(title=rid, xlabel='frames', ylabel='norm.speed and dff')


        clean_axes(f)
        f.tight_layout()
        save_figure(f, sfld / f'{rid}_dff_vs_speed')
    #     break
    # break

# %%
'''
    Make histogram of pearson correlations between filtered variables
'''


speed_th = .2
window = 30

pvals = [[], []]
speed_coeffs, dist_coffs, speed_dist_coeffs, pc_speed_coeffs = [[], []], [[], []], [[], []], [[], []]
# cleaned_sigs = []
for mouse, sess, sessname in track(mouse_sessions, description='Computing pearson corr'):
    if not DO['pearson_corr']: 
        break
    # Get data
    tracking, ang_vel, speed, shelter_distance, dffs, signals, nrois, is_rec, roi_ids = \
                        get_mouse_session_data(mouse, sess, sessions, hanning_window=hanning)

    # Get chunks start end times
    tiff_starts, tiff_ends = get_tiff_starts_ends(is_rec)

    # Get times running in direction
    for n in range(2):
        run_out = np.zeros_like(is_rec)
        if n == 0:
            run_out[derivative(tracking.x) < -speed_th] = 1
        else:
            run_out[derivative(tracking.x) > speed_th] = 1

        keep_frames = run_out * is_rec

        # Slow behav
        _, smooth_speed = get_chunk_rolling_mean_subracted_signal(speed, tiff_starts, tiff_ends, window=window)
        _, smooth_shelter_distance = get_chunk_rolling_mean_subracted_signal(shelter_distance, tiff_starts, tiff_ends, window=window)

        smooth_speed = smooth_speed[keep_frames==1]
        smooth_shelter_distance = smooth_shelter_distance[keep_frames==1]
        # speed_dist_coeffs[n].append((ytrain.sig, model.predict(xtrain)(smooth_speed, smooth_shelter_distance)[0])


        _, dx = get_chunk_rolling_mean_subracted_signal(derivative(tracking.x), tiff_starts, tiff_ends, window=window)
        _, dy = get_chunk_rolling_mean_subracted_signal(derivative(tracking.y), tiff_starts, tiff_ends, window=window)

        dx = dx[keep_frames==1]
        dy = dy[keep_frames==1]

        # loop over rois
        cleaned_sigs = []
        for sig, rid in zip(signals, roi_ids):
            # get chunked dff
            dff = get_chunked_dff(sig, tiff_starts, tiff_ends)

            diff, smoothed = get_chunk_rolling_mean_subracted_signal(dff, tiff_starts, tiff_ends, window=window)
            smoothed = smoothed[keep_frames==1]

            speed_coeffs[n].append(spearmanr(smoothed, smooth_speed)[0])
            pvals[n].append(spearmanr(smoothed, smooth_speed)[1])
            dist_coffs[n].append(spearmanr(smoothed, smooth_shelter_distance)[0])

            if spearmanr(smoothed, smooth_speed)[0] >= .5:
                raise ValueError

            cleaned_sigs.append(smoothed)
        
        # pc = PCA(n_components=1).fit_transform(np.vstack(cleaned_sigs).T)
        # pc_speed_coeffs[n].append(spearmanr(zscore(pc), smooth_speed)[0])


# %%
f, axarr = plt.subplots(ncols=3, figsize=(16, 9), sharey=False, sharex=False)

bins = np.linspace(-1, 1, 30)
bins2 = np.linspace(-1, 1, 10)

axarr[0].hist(speed_coeffs[0], bins=bins, color=[.4, .4, .4], 
        alpha=.5, label='(L) single ROI vs speed', density=True, lw=0, ec='k')
axarr[0].hist(speed_coeffs[1], bins=bins, color=salmon_dark, 
        alpha=.5, label='(R) single ROI vs speed', density=True, lw=0, ec='k')
axarr[0].legend()
axarr[0].axvline(0, ls=':', lw=.5, color=[.2, .2, .2])
axarr[0].set(xlabel='spearman R coeff.', ylabel='density', title='Speed vs DFF correlation')

axarr[1].hist(glm_perfs[0], color=[.4, .4, .4], # bins=np.linspace(0, 2, 30),
        alpha=.5, label='(L) first PC vs speed', density=True, lw=0, ec='k')
axarr[1].hist(glm_perfs[1], color='skyblue', # bins=np.linspace(0, 2, 30),
        alpha=.5, label='(R)first PC vs speed', density=True, lw=0, ec='k')
axarr[1].legend()
axarr[1].axvline(0, ls=':', lw=.5, color=[.2, .2, .2])
axarr[1].set(xlabel='spearman R coeff.',  title='Speed vs 1st PC correlation')

axarr[2].hist(speed_dist_coeffs[0], bins=bins2, color=[.2, .2, .2], 
        alpha=.5, label='speed', density=True, lw=1, ec='k')
axarr[2].set(xlabel='spearman R coeff.', title='Speed vs shelter distance correlation')
axarr[2].axvline(0, ls=':', lw=.5, color=[.2, .2, .2])

clean_axes(f)


# %%
"""
    Fit GLMs to predict a ROIs signal from X,Y speeds
    and compare
"""
def fit_eval_glm(sig, x):
    """
        Tries to predict a roi's signal
        based on the animal's speed using
        cross validated GLM
    """
    # X = pd.DataFrame(dict(x=x))
    X = sm.add_constant(pd.DataFrame(dict(x=x)), prepend=False)

    Y = pd.DataFrame(dict(sig=sig))
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y)

    model = sm.GLM(Y, X).fit(maxiter=1000)
    # performance = mean_squared_error(ytrain.sig, model.predict(xtrain))
    # null_performance = mean_squared_error(ytrain.sig, np.zeros_like(ytrain.sig.values))

    # return np.linalg.norm(ytrain.sig -  model.predict(xtrain)), model, xtest, ytest
    # return mean_squared_error(ytrain.sig, model.predict(xtrain))

    return model.predict(X)

speed_th = .2
window = 30


# cleaned_sigs = []
errs = [[], []]
for mouse, sess, sessname in track(mouse_sessions, description='fitting'):
    if not DO['pearson_corr']: 
        break
    # Get data
    tracking, ang_vel, speed, shelter_distance, dffs, signals, nrois, is_rec, roi_ids = \
                        get_mouse_session_data(mouse, sess, sessions, hanning_window=hanning)

    # Get chunks start end times
    tiff_starts, tiff_ends = get_tiff_starts_ends(is_rec)

    # Slow behav
    _, dx = get_chunk_rolling_mean_subracted_signal(derivative(tracking.x), tiff_starts, tiff_ends, window=window)
    _, dy = get_chunk_rolling_mean_subracted_signal(derivative(tracking.y), tiff_starts, tiff_ends, window=window)
    _, smooth_speed = get_chunk_rolling_mean_subracted_signal(speed, tiff_starts, tiff_ends, window=window)
    _, smooth_shelter_distance = get_chunk_rolling_mean_subracted_signal(shelter_distance, tiff_starts, tiff_ends, window=window)

    dx = dx[is_rec==1]
    dy = dy[is_rec==1]
    smooth_speed= smooth_speed[is_rec==1]
    smooth_shelter_distance=smooth_shelter_distance[is_rec==1]
    
    for sig, rid in zip(signals, roi_ids):
        # get chunked dff
        dff = get_chunked_dff(sig, tiff_starts, tiff_ends)

        _, smoothed = get_chunk_rolling_mean_subracted_signal(dff, tiff_starts, tiff_ends, window=window)
        smoothed = smoothed[is_rec==1]

        # ys.append()
        # x_mse = fit_eval_glm(smoothed, dx)
        # y_mse = fit_eval_glm(smoothed, dy)
        
        # errs[0].append(x_mse)
        # errs[1].append(y_mse)

        f, ax = plt.subplots()
        ax.scatter(smooth_speed, smoothed, alpha=.01)

        # ax.plot(smoothed, lw=2, color='k')
        # ax.plot(fit_eval_glm(np.random.rand(len(smoothed)), smooth_speed))
        # ax.plot(fit_eval_glm(smoothed, smooth_speed))

        # ax.plot(fit_eval_glm(np.random.rand(len(smoothed)), dx))
        # ax.plot(dx)
        # ax.plot(smooth_speed)
        # ax.set(xlim=[0, 9000])
        # break
    break

# f,ax = plt.subplots(figsize=(10, 10))

# ax.scatter(errs[0], errs[1])

# %%

# _, x = get_chunk_rolling_mean_subracted_signal(derivative(tracking.x), tiff_starts, tiff_ends, window=window)
# X = sm.add_constant(pd.DataFrame(dict(speed=x[keep_frames==1])), prepend=False)
# X = pd.DataFrame(dict(speed=smooth_speed))
X = pd.DataFrame(dict(x=dx, y=dy))

Y = pd.DataFrame(dict(sig=smoothed))
xtrain, xtest, ytrain, ytest = train_test_split(X, Y)

model = sm.GLM(ytrain, xtrain[['x']], missing='drop').fit()
model2 = sm.GLM(ytrain, xtrain[['y']], missing='drop').fit()
# model3 = sm.GLM(ytrain, xtrain, missing='drop').fit()

f, ax = plt.subplots()

plt.plot(Y.sig.values, lw=3, color='k')
plt.plot(model.predict(X['x']).values, lw=2,  color='red')
plt.plot(model2.predict(X['y']).values, lw=2,  color='blue')
# plt.plot(model3.predict(X).values, lw=2,  color='m')

ax.set(xlim=[1000, 3000])

# %%
plt.plot(X['x'])
# %%


df = pd.DataFrame(dict(s=smoothed, speed=smooth_speed))
df.head()
# %%
binned = df.groupby(pd.cut(df['speed'], bins=10)).mean()
# %%
plt.plot(binned.speed, binned.s)
# %%
