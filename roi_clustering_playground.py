# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from affinewarp import ShiftWarping, PiecewiseWarping

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

from fcutils.plotting.utils import set_figure_subplots_aspect, clean_axes, save_figure
from fcutils.plotting.plot_elements import plot_mean_and_error
from fcutils.maths.utils import normalise_1d

def show_mtx(mtx, ax=None, f=None):
    if ax is None:
        f, ax = plt.subplots(figsize=(12, 12))
    cax = ax.matshow(mtx, cmap='RdYlGn', )
    f.colorbar(cax, ax=ax, shrink=0.9)

def get_mean_at_shelt_frame(df):
    t = 0
    for mouse in df.mouse.unique():
        t += df.loc[df.mouse == mouse].at_shelt_frame.values[0]
    t  /= len(df.mouse.unique())
    return t

def normalize_time(trace):
        at_s = trace.at_shelt_frame - n_frames_pre
        t = (np.arange(len(trace.signal)) - n_frames_pre) / at_s
        return t


def normalize_time_al_traces(traces):
    T = np.arange(60) - 30
    time_dict = {t:[] for t in T}

    count = []
    for n, (i, trace) in enumerate(traces.iterrows()):

        t = normalize_time(trace)
        t = (t * 10).astype(np.int32)
        sig = trace.signal

        added = []
        for time, s in zip(t, sig):
            if time in added: continue
            if time < -30 or time > 29: continue

            else:
                if time in T:
                    time_dict[time].append(s)
                    added.append(time)
                else:
                    raise ValueError

        for time in T:
            if time not in added:
                time_dict[time].append(np.nan)

    return pd.DataFrame(time_dict)


# %%

# ---------------------------------------------------------------------------- #

#                                    PARAMS                                    #
# ---------------------------------------------------------------------------- #
# fld = Path('D:\\Dropbox (UCL - SWC)\\Project_vgatPAG\\analysis\\doric\\Fede\\plots\\ManualTagsAligned')
fld = Path('/Users/federicoclaudi/Dropbox (UCL - SWC)/Project_vgatPAG/analysis/doric/Fede/plots/ManualTagsAligned')

# load cache
cache = pd.read_hdf(str(fld /  'cached_traces.h5'), key='hdf')
cache = cache.loc[cache.above_noise == True]
n_frames_pre = cache.n_frames_pre.values[0]

# Load correlation mtx
corr = np.load(str(fld / 'cached_corr_mtx.npy'))

if len(cache) != corr.shape[0]:
    raise ValueError('Mismatch between shape of cache and corr mtx, check your params and preprocessing steps')
print(f'Loaded correlation matrix with shape: {corr.shape}')

show_mtx(corr)

# %%

# ---------------------------------------------------------------------------- #
#                                  CLUSTERING                                  #
# ---------------------------------------------------------------------------- #

# -------------------------------- Dendrogram -------------------------------- #

dend = sch.dendrogram(sch.linkage(corr, method='ward'))

# %%

# -------------------------- Hierarchical clustering ------------------------- #

def norm_stack(df, usekey=True, norm=True):
    sigs = []
    for i,t in df.iterrows():
        if usekey:
            sig = t.signal
            sig = sig

        else:
            sig = t.values

        if norm:
            sigs.append(normalise_1d(sig))
        else:
            sigs.append(sig)

    signals = np.vstack(sigs)
    return signals

# FIT THE CLUSTERING ALGORITHM
N_CLUSTERS = 4
WARP = False
SHOW_HEATMAPS = False

cluster = AgglomerativeClustering(n_clusters=N_CLUSTERS, compute_full_tree=True,
                                affinity='euclidean', linkage='complete')
_ = cluster.fit_predict(corr)


model = ShiftWarping(maxlag=.1, smoothness_reg_scale=20.)

# # PLOT
# f, axarr = plt.subplots(nrows=4, ncols=30, figsize=(80, 12))
# # axarr = axarr.flatten()
# for ax in axarr.flatten(): ax.axis('off')

# f.suptitle(f'N clusters: {N_CLUSTERS} - time warping: {WARP}')

# # Plot each cluster's traces
# for ax, clust_id in zip(axarr, set(cluster.labels_)):
#     idxs = np.argwhere(cluster.labels_ == clust_id)
#     clust_sigs = cache.iloc[idxs.ravel()]

#     # time warping
#     if WARP:
#         signals = norm_stack(clust_sigs)

#         tw_sigs = signals[np.newaxis,:, :].transpose((1, 2, 0))
#         model.fit(tw_sigs, iterations=50, warp_iterations=200)
#         warped =  model.transform(tw_sigs).squeeze()

#     else:
#         warped = norm_stack(clust_sigs, norm=False)

#         df = normalize_time_al_traces(clust_sigs)
#         cluster_mean = df.mean()

#         for roin, roi in enumerate(clust_sigs.roi_n.unique()):
#             ax = axarr[clust_id, roin]
#             df = normalize_time_al_traces(clust_sigs.loc[clust_sigs.roi_n == roi])

#             for i, r in df.iterrows():
#                 ax.plot(r, color='k', lw=1, alpha=.45)
#             ax.plot(df.mean(), lw=4, color='r')
#             ax.plot(cluster_mean, lw=4, color='b')


#             ax.axis('on')


# %%

model = ShiftWarping(maxlag=.05, smoothness_reg_scale=20.)

# PLOT
f, axarr = plt.subplots(ncols=3, nrows=2,figsize=(20, 12))
axarr = axarr.flatten()

f.suptitle(f'N clusters: {N_CLUSTERS} - time warping: {WARP}')

# Plot each cluster's traces
for ax, clust_id in zip(axarr, set(cluster.labels_)):
    idxs = np.argwhere(cluster.labels_ == clust_id)
    clust_sigs = cache.iloc[idxs.ravel()]

    # time warping
    if WARP:
        signals = norm_stack(clust_sigs, norm=False)

        tw_sigs = signals[np.newaxis,:, :].transpose((1, 2, 0))
        model.fit(tw_sigs, iterations=50, warp_iterations=200)
        warped =  model.transform(tw_sigs).squeeze()
    else:
        warped = norm_stack(clust_sigs, norm=False)
        
        # df = normalize_time_al_traces(clust_sigs)
        # for i, trace in df.iterrows():
        #     ax.plot(trace, color='k', lw=.4)
        # ax.plot(df.mean(), lw=3, color='r')          
        # ax.axvline(0, lw=4, color='g')
        # ax.axvline(10, lw=4, color='m')

    if SHOW_HEATMAPS:
        ax.imshow(warped, cmap='bwr', vmin=-3, vmax=3)
    else:
        ax.plot(warped.T, color='k', alpha=.4)
        ax.plot(np.mean(warped.T, 1), color='r', lw=3)

    ax.axvline(120, lw=4, color='g')
    ax.axvline(cache.at_shelt_frame.mean(), lw=4, color='m')

    ax.set(title=f'Cluster {clust_id} | {warped.shape[0]} trials')
        
    # ax.set(ylim=[-50, 50])
clean_axes(f)
f.tight_layout()
# save_figure(f, str(fld / 'clusters_traces'))

# %%
# Plot correlation matrix (sorted)

srtd = np.sort(cluster.labels_)
changes = [np.where(srtd == n)[0][0] for n in range(N_CLUSTERS)]

sort_idx = np.argsort(cluster.labels_)

corr_clust = corr.copy()
corr_clust[:] = corr_clust[:, sort_idx]
corr_clust[:] = corr_clust[sort_idx, :]
clean_axes(f)

fig, ax = plt.subplots(figsize=(12, 12))
cax = ax.matshow(corr_clust, cmap='RdYlGn', )
fig.colorbar(cax, ax=ax, shrink=0.9)

for change in changes:
    ax.axvline(change, lw=2, color='m')
    ax.axhline(change, lw=2, color='m')

nc = cache.copy()
nc = nc.set_index(sort_idx).sort_index()




# %%

cache['cluster'] = cluster.labels_

single_cluster_clust_id = []
n_clust_per_roi = []
for mouse in cache.mouse.unique():
    mouse_c = cache.loc[cache.mouse == mouse]

    for sess in mouse_c.session.unique():
        sess_c = mouse_c.loc[mouse_c.session == sess]

        for roi in sess_c.roi_n.unique():
            roic = sess_c.loc[sess_c.roi_n == roi]
            n_clusts = len(roic.cluster.unique())
            n_clust_per_roi.append(n_clusts)

            if n_clusts == 1:
                if len(roic) == 1: 
                    continue
                single_cluster_clust_id.append(roic.cluster.unique()[0])
        
            # f, axarr = plt.subplots(ncols=4, sharex=True, sharey=True, figsize=(20, 5))
            # colors  = dict(zip(roic.cluster.unique(), 'rgbm'))
            # axes  = dict(zip(roic.cluster.unique(), axarr))

            # for i,r in roic.iterrows():
            #     axes[r.cluster].plot(r.signal, color=colors[r.cluster], lw=2, alpha=.5)
            # for ax in axarr: 
            #     ax.axhline(0, color='k')

            # save_figure(f, str(fld / f'{mouse}_{sess}_{roi}'))
            # del f
    

f, ax = plt.subplots()
_ = ax.hist(n_clust_per_roi)


# %%

# TODO look at distribution of clusters across ROI for each trial in a single mouse at the time
# %%
forv = cache.loc[(cache.mouse == 'BF164p1')&(cache.session=='19JUN05')&(cache.roi_n==7)]
sigs = [v.signal for i,v in forv.iterrows()]

cm = np.corrcoef(sigs)

f, axarr = plt.subplots(ncols=2, figsize=(16, 9))

show_mtx(cm, ax=axarr[0], f=f)


from brainrender.colors import colorMap

colors = dict(zip([0, 1, 2, 3, 4, 5, 6], 'grgmk'))

for n, s in enumerate(sigs):
    axarr[1].plot(s, color=colors[forv.iloc[n].cluster], 
        lw=5, alpha=.4, label=f'trial: {n} cluster {forv.iloc[n].cluster}')
axarr[1].legend()

axarr[1].axvline(120, color='k')

# %%
