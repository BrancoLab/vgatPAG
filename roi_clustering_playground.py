# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

from fcutils.plotting.utils import calc_nrows_ncols, set_figure_subplots_aspect, clean_axes, save_figure
from fcutils.plotting.plot_elements import plot_mean_and_error

def show_mtx(mtx, ax=None, f=None):
    if ax is None:
        f, ax = plt.subplots(figsize=(12, 12))
    cax = ax.matshow(mtx, cmap='RdYlGn', )
    f.colorbar(cax, ax=ax, shrink=0.9)


# %%

# ---------------------------------------------------------------------------- #
#                                    PARAMS                                    #
# ---------------------------------------------------------------------------- #
fld = Path('D:\\Dropbox (UCL - SWC)\\Project_vgatPAG\\analysis\\doric\\Fede\\plots\\ManualTagsAligned')

# load cache
cache = pd.read_hdf(str(fld /  'cached_traces.h5'), key='hdf')
n_frames_pre = cache.n_frames_pre.values[0]

# Load correlation mtx
corr = np.load(str(fld / 'cached_corr_mtx.npy'))
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
# FIT THE CLUSTERING ALGORITHM
N_CLUSTERS = 5
cluster = AgglomerativeClustering(n_clusters=N_CLUSTERS, 
                                affinity='euclidean', linkage='complete')
_ = cluster.fit_predict(corr)


# PLOT
f, axarr = plt.subplots(ncols=3, nrows=2, figsize=(16, 12), sharex=True, sharey=True)
axarr = axarr.flatten()

# Plot each cluster's traces
for ax, clust_id in zip(axarr, set(cluster.labels_)):
    idxs = np.argwhere(cluster.labels_ == clust_id)
    clust_sigs = cache.iloc[idxs.ravel()]

    sigs = []
    for i,t in clust_sigs.iterrows():
        sigs.append(t.signal - np.mean(t.signal[:n_frames_pre-1]))
    signals = np.vstack(sigs)
    ax.plot(signals.T, color='k', lw=1, alpha=.5)

    mean, std = np.mean(signals, 0), np.std(signals, 0)
    plot_mean_and_error(mean, std, ax, zorder=99, color='r')

    ax.axvline(n_frames_pre, lw=2, color='g')
    ax.set(title=f'Cluster {clust_id} - {len(clust_sigs)} trials', ylim=[-150, 150])


# Plot correlation matrix (sorted)
srtd = np.sort(cluster.labels_)
changes = [np.where(srtd == n)[0][0] for n in range(N_CLUSTERS)]

corr_clust = corr.copy()
corr_clust[:] = corr_clust[:, np.argsort(cluster.labels_)]
corr_clust[:] = corr_clust[np.argsort(cluster.labels_), :]
axarr[-1].axis('off')
clean_axes(f)

fig, ax = plt.subplots(figsize=(12, 12))
cax = ax.matshow(corr_clust, cmap='RdYlGn', )
fig.colorbar(cax, ax=ax, shrink=0.9)

for change in changes:
    ax.axvline(change, lw=2, color='m')
    ax.axhline(change, lw=2, color='m')




# %%
