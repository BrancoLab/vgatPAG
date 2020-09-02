# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
from brainrender.colors import makePalette, colorMap
import numpy as np
from scipy.signal import medfilt
from pathlib import Path
from palettable.cmocean.sequential import Matter_8 as CMAP
from palettable.cmocean.sequential import Speed_8 as CMAP2
from palettable.cmocean.sequential import Deep_8 as CMAP3
from matplotlib.lines import Line2D
from tqdm import tqdm
from scipy.stats.stats import pearsonr   

import scipy
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import normalize


from fcutils.plotting.utils import calc_nrows_ncols, set_figure_subplots_aspect, clean_axes, save_figure
from fcutils.plotting.plot_elements import plot_mean_and_error

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
)
from Analysis.tag_aligned import (
    manual_tags,
    get_tags_by,
)


# %%

# ---------------------------------------------------------------------------- #
#                                 CACHE SIGNALS                                #
# ---------------------------------------------------------------------------- #

""" 
    Store each ROIs' signal for each trial in a dataframe
"""

fld = Path('D:\\Dropbox (UCL - SWC)\\Project_vgatPAG\\analysis\\doric\\Fede\\plots\\ManualTagsAligned')

CACHE_SIGNALS = False

def process_sig(sig, start, end, n_sec_post, norm=False, filter=True):
    if filter: # median filter
        sig = medfilt(sig, kernel_size=5)
    if norm: # normalize with baseline
        baseline = np.mean(sig[start: start + n_frames_pre - 1])
        sig =  sig - baseline
    return sig


fps = 30
n_sec_pre = 4
n_sec_post = 4
n_frames_pre = n_sec_pre * fps
n_frames_post = n_sec_post * fps


if CACHE_SIGNALS:
        
    cache = dict(
        mouse = [],
        session = [],
        session_frame = [],
        tag_type = [],
        n_frames_pre = [],
        n_frames_post = [],
        roi_n = [],
        signal = [],
    )


    tag_type = 'VideoTag_B'
    event_type =  ['Loom_Escape', 'US_Escape', 'LoomUS_Escape'] #  

    NORMALIZE = True # if true divide sig by the mean of the sig in n_sec_pre
    FILTER = True # if true data are median filtered to remove artefact

    
    for mouse, sess, sessname in mouse_sessions:
        # if sessname not in include_sessions: continue
        print(f'Processing {sessname}\n')

        # Get data
        tracking, ang_vel, speed, shelter_distance, signals, nrois, is_rec = get_mouse_session_data(mouse, sess, sessions)
        tags = get_tags_by(mouse=mouse, sess_name=sess, event_type=event_type, tag_type=tag_type)
        for count, (i, tag) in enumerate(tags.iterrows()):
            start = tag.session_frame - n_frames_pre
            end = tag.session_frame + n_frames_post

            if np.sum(is_rec[start:end]) == 0: # recording was off
                continue

            # CACHE
            for roin, sig in enumerate(signals):
                if np.std(sig[start:end]) == 0:
                    raise ValueError
                sig = process_sig(sig, start, end, n_sec_pre, norm=NORMALIZE, filter=FILTER)

                cache['mouse'].append(mouse)
                cache['session'].append(sess)
                cache['session_frame'].append(tag.session_frame)
                cache['tag_type'].append(tag_type)
                cache['n_frames_pre'].append(n_frames_pre)
                cache['n_frames_post'].append(n_frames_post)
                cache['roi_n'].append(roin)
                cache['signal'].append(sig[start:end])

    cache = pd.DataFrame(cache)
    cache.to_hdf(os.path.join(fld, 'cached_traces.h5'), key='hdf')
else:
    cache = pd.read_hdf(os.path.join(fld, 'cached_traces.h5'), key='hdf')
cache.head()

# %%

# ----------------------- Compute cross correlation mtx ---------------------- #

COMPUTE = False
if COMPUTE:

    n_sigs = len(cache)

    corr = np.zeros((n_sigs, n_sigs))

    done = []
    for i in tqdm(range(n_sigs)):
        for j in range(n_sigs):
            if (i, j) in done or (j, i) in done: 
                continue
            else:
                done.append((i, j))

            if i == j:
                corr[i, j] = 1.
            else:
                _corr = pearsonr(cache.iloc[i]['signal'], cache.iloc[j]['signal'])[0]
                corr[i, j] = _corr
                corr[j, i] = _corr
    np.save(os.path.join(fld, 'cached_corr_mtx.npy'), corr)
else:
    corr = np.load(os.path.join(fld, 'cached_corr_mtx.npy'))

    
plt.imshow(corr)



# %%


d = sch.distance.pdist(corr)   # vector of ('55' choose 2) pairwise distances
L = sch.linkage(d, method='complete')
ind = sch.fcluster(L, 0.5*d.max(), 'distance')
# columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
# df = df.reindex_axis(columns, axis=1)

corr_clust = corr.copy()
corr_clust[:] = corr_clust[:, np.argsort(ind)]
corr_clust[:] = corr_clust[np.argsort(ind), :]

fig, ax = plt.subplots(figsize=(12, 12))
cax = ax.matshow(corr_clust, cmap='RdYlGn')


# %%
dend = sch.dendrogram(sch.linkage(corr, method='ward'))
# %%
from sklearn.cluster import AgglomerativeClustering
N_CLUSTERS = 5
cluster = AgglomerativeClustering(n_clusters=N_CLUSTERS, affinity='euclidean', linkage='ward')
_ = cluster.fit_predict(corr)

# %%
f, axarr = plt.subplots(ncols=3, nrows=2, figsize=(16, 12), sharex=True, sharey=True)
axarr = axarr.flatten()

for ax, clust_id in zip(axarr, set(cluster.labels_)):
    idxs = np.argwhere(cluster.labels_ == clust_id)
    clust_sigs = cache.iloc[idxs.ravel()]

    sigs = []
    for i,t in clust_sigs.iterrows():
        # min, max = t.signal.min(), t.signal.max()
        # sigs.append((t.signal-min)/(max-min))
        sigs.append(t.signal - np.mean(t.signal[:n_frames_pre-1]))
    signals = np.vstack(sigs)

    ax.plot(signals.T, color='k', lw=1, alpha=.5)

    mean, std = np.mean(signals, 0), np.std(signals, 0)
    # plot_mean_and_error(mean, std, ax, zorder=99, color='r')

    ax.axvline(n_frames_pre, lw=2, color='g')
    ax.set(title=f'Cluster {clust_id} - {len(clust_sigs)} trials', ylim=[-150, 150])


srtd = np.sort(cluster.labels_)
changes = [np.where(srtd == n)[0][0] for n in range(N_CLUSTERS)]

corr_clust = corr.copy()
corr_clust[:] = corr_clust[:, np.argsort(cluster.labels_)]
corr_clust[:] = corr_clust[np.argsort(cluster.labels_), :]
axarr[-1].axis('off')
clean_axes(f)

# fig, ax = plt.subplots(figsize=(12, 12))
# cax = ax.matshow(corr_clust, cmap='RdYlGn', )
# fig.colorbar(cax, ax=ax, shrink=0.9)

# for change in changes:
#     ax.axvline(change, lw=2, color='m')
#     ax.axhline(change, lw=2, color='m')

# %%
