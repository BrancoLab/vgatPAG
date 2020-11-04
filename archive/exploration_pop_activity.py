# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
from random import choices
import palettable
from sklearn.linear_model import LogisticRegression

from Analysis  import (
        mice,
        sessions,
        recordings,
        recording_names,
        stimuli,
        clean_stimuli,
        get_mouse_session_data,
        sessions_fps,
        mouse_sessions,
        pxperframe_to_cmpersec,
        compute_stationary_vs_locomoting,
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
from fcutils.plotting.colors import *
from fcutils.modelling.svm import fit_svc_binary, plot_svc_boundaries, plot_svc_boundaries_multiclass

from brainrender.colors import makePalette, colorMap


from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import statsmodels.api as sm



from vgatPAG.paths import output_fld
output_fld = Path(output_fld)
save_fld = output_fld / 'baseline'


# %%
# Get explorations
"""
    Get population activity and behaviour during the exploration of each experiment
"""
explorations = {}
for mouse, sess, name in tqdm(mouse_sessions):
    stims = stimuli[name]

    if not stims.all:
        explorations[name] = None
        continue

    tracking, ang_vel, speed, shelter_distance, signals, nrois, is_rec = get_mouse_session_data(mouse, sess, sessions)

    end = stims.all[0]
    data = tracking[:end]
    data['s'] = speed[:end]
    data['is_rec'] = is_rec[:end]
    data['shelt_dist'] = shelter_distance[:end]

    in_shelt = np.zeros_like(shelter_distance)
    in_shelt[shelter_distance <= 0] = 1
    data['in_shelter'] = in_shelt[:end]

    data['is_locomoting'] = compute_stationary_vs_locomoting(speed)[:end]

    for n, signal in enumerate(signals):
        data[f'roi_{n}'] = signal[:end]

    explorations[name] = data


# %%
# Fit PCAs

"""
    Fit a PCA to population activity for each exploration, only using frames
    while the doric recording is on.
"""
pcas = {}
pcas_ca_data = {}
pcas_transformed = {}
for mouse, sess, name in tqdm(mouse_sessions):
    if explorations[name] is None:
        pcas[name] = None
        pcas_ca_data[name] = None
        pcas_transformed[name] = None
        continue

    data = explorations[name].loc[explorations[name].is_rec == 1.]
    roi_cols = [c for c in data.columns if 'roi_' in c]
    ca_data = data[roi_cols].values

    pcas[name] = PCA(n_components=2).fit(ca_data)
    pcas_ca_data[name] = ca_data
    pcas_transformed[name] = pcas[name].transform(ca_data)

# %%
# PCA visualise variance explained
f, axarr = plt.subplots(ncols=3, nrows=3, figsize=(16, 16))
axarr = axarr.flatten()
s = 0

for mouse, sess, name in tqdm(mouse_sessions):
    if explorations[name] is None:
        axarr[s].axis('off')
        s += 1
        continue

    data = explorations[name].loc[explorations[name].is_rec == 1.]
    roi_cols = [c for c in data.columns if 'roi_' in c]
    ca_data = data[roi_cols].values

    pca = PCA(n_components=8).fit(ca_data)

    ratios = pca.explained_variance_ratio_

    axarr[s].plot(np.arange(8), pca.explained_variance_ratio_, 'o-', lw=2, color='k',)
    axarr[s].set(title=name, xlabel='PCs', ylabel='explained variance ratio')
    s += 1

clean_axes(f)


# %%
# Plot SVM predictions
"""
    Use SVM to predict stuff from population activity during exploration
"""

def divide_data_by(by, data, N):
    if by == 'shelter':
        A = data.loc[data.in_shelter == 1]
        B = data.loc[data.in_shelter == 0]
        A_label, B_label = 'IN', 'OUT'
    elif by == 'xpos':
        A = data.loc[(data.x >=400)&(data.x < 600)]
        B = data.loc[(data.x >=800)&(data.x < 1000)]
        A_label, B_label = '400 < x < 600', '600 < x < 800'

    else:
        raise ValueError

    if len(A) > N:
        A = A.sample(N)
        A_pc = pcas[name].transform(A[roi_cols].values)
    else:
        A, A_pc = None, None

    if len(B) > N:
        B = B.sample(N)
        B_pc = pcas[name].transform(B[roi_cols].values)
    else:
        B, B_pc = None, None

    return A, A_pc, B, B_pc, A_label, B_label


"""
    And overlay samples from different categories (e.g. in/out shelter)
"""

KEEP_ONLY_STATIONARY = True # If False only fast is kept if None all is kept
SHUFFLE = False # randomize labels for testing
PREDICT = 'shelter'


f, axarr = plt.subplots(ncols=3, nrows=3, figsize=(16, 16))
axarr = axarr.flatten()
s = 0
f.suptitle(f'Shelter representation')

overal_accuracies = {}
for mouse, sess, name in tqdm(mouse_sessions):
    if pcas[name] is None: continue

    # Plot KDE of whole PCA trace
    sns.kdeplot(pcas_transformed[name][:, 0], pcas_transformed[name][:, 1], alpha=.1, cmap='gray', lw=0,
                shade=True, ax=axarr[s], shade_lowest=False, n_levels=20)

    # Prep data
    N = 500
    data = explorations[name].loc[explorations[name].is_rec == 1.]
    roi_cols = [c for c in data.columns if 'roi_' in c]

    if KEEP_ONLY_STATIONARY:
        data = data.loc[data.is_locomoting == 0]


    # Get in/out of shelter times and put through PCA 
    A, A_pc, B, B_pc, A_label, B_label = divide_data_by(PREDICT, data, N)   
    
    if A_pc is not None:
        axarr[s].scatter(A_pc[:, 0], A_pc[:, 1], color=shelt_dist_color, zorder=99, alpha=.4, s=25, label=A_label)
    if B_pc is not None:
        axarr[s].scatter(B_pc[:, 0], B_pc[:, 1], color=salmon, zorder=99, alpha=.4, s=25, label=B_label)
    


    # Fit a SVM to predict when mouse in shelter
    if A is not None and B is not None:
        if len(A_pc) != len(B_pc):
            raise ValueError

        X = np.vstack([A_pc, B_pc])
        y = np.zeros(len(X))
        y[:len(A_pc)] = 1

        if SHUFFLE:
            np.random.shuffle(y)

        accuracy, svc = fit_svc_binary(X, y)
        plot_svc_boundaries(axarr[s], svc)
        axarr[s].set(title=name+f' svm accuracy: {round(accuracy, 2)}\n avg runn speed = {round(data.s.mean(), 2)}\n nrois:{len(roi_cols)}', 
                        xlabel='PC1', ylabel='PC2')

        overal_accuracies[name] = accuracy

    # Clean axes
    axarr[s].legend()
    s += 1

    # break
# 

axarr[-1].axis('off')

clean_axes(f)
set_figure_subplots_aspect(hspace=.6, top=.9, wspace=.25)
save_figure(f, str(save_fld / f'Shelter representation {"stationary" if KEEP_ONLY_STATIONARY else ""} {PREDICT}'))

# %%
# Fit SVM to individual ROIs
f, axarr = plt.subplots(ncols=3, nrows=3, figsize=(16, 16))
axarr = axarr.flatten()
s = 0
f.suptitle(f'individual ROI SVM accuracy (shelter detection)')


for mouse, sess, name in tqdm(mouse_sessions):
    if pcas[name] is None: continue

    # Prep data
    N = 500
    data = explorations[name].loc[explorations[name].is_rec == 1.]
    roi_cols = [c for c in data.columns if 'roi_' in c]

    if KEEP_ONLY_STATIONARY:
        data = data.loc[data.is_locomoting == 0]

    # split between in/out shelter
    A = data.loc[data.in_shelter == 1]
    B = data.loc[data.in_shelter == 0]
    A_label, B_label = 'IN', 'OUT'


    # for each ROI
    accuracies = []
    for rn, roic in enumerate(roi_cols):
        A_sig = A[roic].sample(N).values
        B_sig = B[roic].sample(N).values

        # Crete data and labels arrays
        X = np.hstack([A_sig, B_sig]).reshape(-1, 1)
        y = np.zeros(len(X))
        y[:len(A_pc)] = 1

        accuracy, svc = fit_svc_binary(X, y, max_iter=10000000)
        accuracies.append(accuracy)

    axarr[s].bar(np.arange(len(accuracies)), accuracies, color='k')
    axarr[s].set(title=name + f'\n Pop. level SVM, accuracy: {round(overal_accuracies[name], 2)}', 
                xlabel='ROI', 
                xticklabels=[r.split('_')[-1] for r in roi_cols],
                xticks=np.arange(len(accuracies)),
                ylabel='accuracy', ylim=[0,1])
    axarr[s].axhline(.5, ls='--', lw=1, zorder=99, color='r')
    axarr[s].axhline(overal_accuracies[name], lw=5, zorder=90, color='w')
    axarr[s].axhline(overal_accuracies[name], ls='--', lw=3, zorder=99, color='g')
    s += 1

axarr[-1].axis('off')

clean_axes(f)
set_figure_subplots_aspect(hspace=.6, top=.92, wspace=.25)
save_figure(f, str(save_fld / f'Shelter representation single ROIs'))


# %%
# For each ROI plot the mean activity when the mouse is in or out of the shelter
f, axarr = plt.subplots(ncols=3, nrows=3, figsize=(16, 16))
axarr = axarr.flatten()
s = 0
f.suptitle(f'individual ROI SVM accuracy (shelter detection)')

fld = Path('/Users/federicoclaudi/Dropbox (UCL - SWC)/Project_vgatPAG/analysis/doric/Fede/plots/baseline/SINGLE_ROI_TRACES')

deltas = []
for mouse, sess, name in tqdm(mouse_sessions):
    if pcas[name] is None: continue

    # Prep data
    data = explorations[name].loc[explorations[name].is_rec == 1.]
    roi_cols = [c for c in data.columns if 'roi_' in c]

    if KEEP_ONLY_STATIONARY:
        data = data.loc[data.is_locomoting == 0]

    # split between in/out shelter
    A = data.loc[data.in_shelter == 1]
    B = data.loc[data.in_shelter == 0]
    N = np.min([len(A), len(B)])
    A_label, B_label = 'IN', 'OUT'


    # for each ROI
    accuracies = []
    for rn, roic in enumerate(roi_cols):
        A_sig = np.median(A[roic].sample(N).values)
        B_sig = np.median(B[roic].sample(N).values)

        # save a figure with single ROI's traces
        # f2, ax = plt.subplots(figsize=(12, 8))
        # ax.scatter(A[roic].index.values, A[roic].values, color='r', label=A_label, s=2, alpha=.8)
        # ax.scatter(B[roic].index.values, B[roic].values, color='b', label=B_label, s=2, alpha=.8)
        # ax.axhline(A_sig, color='r')
        # ax.axhline(B_sig, color='b')
        # ax.legend()
        # ax.set(title=name+'_'+roic, xlabel='Frames', ylabel='signal')
        # clean_axes(f2)
        # save_figure(f2, str(fld / (name+'_'+roic)), verbose=False)
        # plt.close(f2)

        # save a figure with logistic regression
        X = np.hstack([A[roic].values, B[roic].values]).reshape(-1, 1)
        y = np.hstack([np.ones(len(A[roic])), np.zeros(len(B[roic]))])
        lr = LogisticRegression().fit(X, y.ravel())
        f2, ax2 = plt.subplots(figsize=(12, 8))
        ax2.scatter(A[roic].values, [1 for i in A[roic]], color='r', label=A_label, s=5, alpha=.8, zorder=99)
        ax2.scatter(B[roic].values, [0 for i in B[roic]], color='b', label=B_label, s=5, alpha=.8, zorder=99)
        ax2.scatter(X, lr.predict_proba(X)[:, 1], s=1, alpha=.8)
        clean_axes(f2)
        save_figure(f2, str(fld / (name+'_'+roic+'logreg')), verbose=False)
        plt.close(f2)

        x = [rn-.1, rn+.1]
        axarr[s].plot(x, [A_sig, B_sig], lw=2, color='k')
        axarr[s].scatter(x, [A_sig, B_sig], 
                            s=50, c=['r', 'b'], linewidths=2, edgecolors ='k', zorder=99)

        deltas.append(A_sig - B_sig)


    axarr[s].set(
            title=name, 
            xlabel='ROI',
            xticklabels=[r.split('_')[-1] for r in roi_cols],
            xticks=np.arange(len(roi_cols)),
            ylabel='red in blue out'
            # ylabel=r'$\frac{IN - OUT}{IN + OUT}$'
            )

    s += 1
_ = axarr[-1].hist(deltas, bins=20, density=True)
_ = axarr[-1].set(xlim=[-100, 100])
# %%
# Fit a SVM to predict X position in 3 bins

f, axarr = plt.subplots(ncols=3, nrows=3, figsize=(16, 16))
axarr = axarr.flatten()
s = 0

for mouse, sess, name in tqdm(mouse_sessions):
    if pcas[name] is None: continue

    # prep data
    N = 250
    data = explorations[name].loc[explorations[name].is_rec == 1.]
    # data = data.loc[data.is_locomoting == 0]
    
    try:
        dfs = [
            data.loc[(data.shelt_dist <= 0)].sample(N),
            data.loc[(data.shelt_dist > 200)&(data.shelt_dist <= 400)].sample(N),
            data.loc[(data.shelt_dist > 500)&(data.shelt_dist <= 700)].sample(N),
        ]
    except:
        print(f'Not enough samples for {name}')
        axarr[s].axis('off')
        s += 1
        continue

    if len(set([len(df) for df in dfs])) != 1: raise ValueError
    
    for n, df in enumerate(dfs):
        df['label'] = [n for i in np.arange(N)]

    roi_cols = [c for c in data.columns if 'roi_' in c]
    
    X = np.vstack([df[roi_cols].values for df in dfs])
    X = pcas[name].transform(X)
    
    y = np.concatenate([df['label'] for df in dfs])

    # Fit
    accuracy, svc = fit_svc_binary(X, y, max_iter=-1)
    
    # Visualise
    axarr[s].set(title = f'{name} - acc.:{round(accuracy, 2)} - chance:{round(1/len(dfs), 2)}')

    colors = palettable.cmocean.sequential.Algae_5.hex_colors[:len(dfs)]
    for df, color in zip(dfs, colors):
        label = f'Max s.dist.: {int(df.shelt_dist.max())}'
        X = pcas[name].transform(df[roi_cols].values)
        axarr[s].scatter(X[:, 0], X[:, 1], c=color, lw=.5, edgecolors ='k', label=label)

    plot_svc_boundaries_multiclass(axarr[s], svc, cmap = palettable.cmocean.sequential.Algae_5.mpl_colormap)
    axarr[s].legend()
    s +=1 
    
    
    # break


axarr[-1].axis('off')

clean_axes(f)
set_figure_subplots_aspect(hspace=.6, top=.9, wspace=.25)
save_figure(f, str(save_fld / f'Distance representation'))






# %%
# Plot time spent in shetler across mice
f, axarr = plt.subplots(ncols=1, nrows=2, figsize=(16, 9))


s = 0
tot_time_inshelt = {}
for mouse in mice:
    for sess in tqdm(sessions[mouse]):
        name = f'{mouse}-{sess}'
        if pcas[name] is None: continue

        in_shelt = explorations[name].in_shelter.values
        tot_time_inshelt[name] = np.sum(in_shelt)

        axarr[1].plot(np.cumsum(in_shelt), label=name)
        axarr[1].legend()

axarr[0].bar(np.arange(len(tot_time_inshelt.keys())), tot_time_inshelt.values(), color='k')
_ = axarr[0].set(xticks=np.arange(len(tot_time_inshelt.keys())), xticklabels=tot_time_inshelt.keys())
axarr[0].set(ylabel='tot_time_in_shelter', xlabel='session')
axarr[1].set(ylabel='comulative time in shelter', xlabel='frames')

# %%
# Plot general population activity in vs out of the shelter
f, axarr = plt.subplots(ncols=3, nrows=3, figsize=(16, 16))
axarr = axarr.flatten()
s = 0
f.suptitle(f'Shelter representation')

for mouse, sess, name in tqdm(mouse_sessions):
    if pcas[name] is None: continue

    data = explorations[name].loc[explorations[name].is_rec == 1.]
    roi_cols = [c for c in data.columns if 'roi_' in c]

    in_shelt = data.loc[data.in_shelter == 1]
    out_shelt = data.loc[data.in_shelter == 0]

    for roi in roi_cols:
        bar = in_shelt[roi].mean()
        axarr[s].plot([0, 1], [in_shelt[roi].mean()/bar, out_shelt[roi].mean()/bar], 'o-', alpha=.5, ms=20, lw=3)

    axarr[s].axhline(1, ls='--', lw=2, color=[.6, .6, .6])
    axarr[s].set(xticks=[0, 1], xticklabels=['IN', 'OUT'], title = name, xlim=[-.2, 1.2])
    s += 1

clean_axes(f)

