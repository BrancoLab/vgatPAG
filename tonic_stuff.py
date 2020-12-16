import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from brainrender._colors import map_color
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pyinspect import install_traceback
install_traceback()

from fcutils.plotting.utils import calc_nrows_ncols, clean_axes, save_figure
from fcutils.maths.utils import rolling_mean, derivative
from vgatPAG.database.db_tables import ManualBehaviourTags, Roi, Sessions

from Analysis import get_session_data, get_session_tags, get_tags_sequences, get_active_rois, seq_type

# %%

pre_pos_s = 1.5
M = int(pre_pos_s*30)
lbls = ('stim', 'start', 'run', 'shelter', 'stop')

xlbl = dict(
        xlabel='time from tag\n(s)', 
        xticks=[0, M, 2*M], 
        xticklabels=[-pre_pos_s, 0, pre_pos_s]
        )


fld = Path('D:\\Dropbox (UCL)\\Project_vgatPAG\\analysis\\doric\\Fede\\\ddf_tag_aligned_all_tags_V2')

def dff(sig):
    th = np.nanpercentile(sig[:(3 * 30)], 30)
    return rolling_mean((sig - th)/th, 3)


# %%
f, axarr = plt.subplots(ncols=3, nrows = 2, figsize=(16, 12))
axarr = axarr.flatten()
outcomes = dict(complete=[], failed=[], aborted=[], incomplete=[])
S = len(Sessions.fetch())
for nsess, sess in enumerate(Sessions.fetch(as_dict=True)):
    print(sess['mouse'], sess['date'])
    color = map_color(nsess, 'viridis', 0, S)


    data, rois = get_session_data(sess['mouse'], sess['date'], roi_data_type='raw')
    if data is None: 
        continue

    tags = get_session_tags(sess['mouse'], sess['date'], 
                        etypes=('visual', 'audio', 'audio_visual'), 
                        ttypes=('A' , 'H', 'B', 'C', 'E', 'D'))
    
    # get tags sequences
    sequences = get_tags_sequences(tags)
    colors = [map_color(n, 'viridis', 0, len(sequences)) for n in range(len(sequences))]

    # Get which ROIs show escape-related acivity
    active_rois = get_active_rois(rois, sequences, sess)

    # fit pca
    signals = active_rois.values[data.is_rec==1, :].astype(np.float32)
    scaler = StandardScaler().fit(signals)
    pca = PCA(n_components=1).fit(scaler.transform(signals))
    pc = pca.transform(scaler.transform(active_rois.values))

    # iterate sequences
    prev_stim = 0
    for seq in sequences:
        if seq.STIM - prev_stim < 60*30:
            continue
        else:
            pre_stim = seq.STIM

        start = seq.STIM - int(3 * 30)
        if data.is_rec[start] != 0:
            if seq_type(seq) != 'complete':
                continue

            # Compute mean DFF threshold
            ths, maxes = [], []
            for roi in active_rois.columns:
                th = np.nanpercentile(active_rois[roi][start:seq.STIM], 30)
                ths.append(th)
                maxes.append(dff(active_rois[roi][start:seq.E]))
            TH = np.mean(ths)
            # TH = np.mean(pc[start:seq.STIM])

            outcomes[seq_type(seq)].append(TH)
            if seq_type(seq) == 'complete':
                color = [.4 , .4, .4]
            elif seq_type(seq) == 'failed':
                color = 'r'
            elif seq_type(seq) == 'incomplete':
                continue
            else:
                color = 'b'

            if  seq_type(seq) not in ('failed', 'incomplete'):
                if np.max(data.s[seq.STIM:seq.E]) > 200:
                    continue
                axarr[0].scatter(TH, (seq.A-seq.STIM)/30, s=100, color=color, alpha=.5, lw=1, edgecolors=[.4, .4, .4])
                axarr[1].scatter(TH, (seq.H-seq.STIM)/30, s=100, color=color, alpha=.5, lw=1, edgecolors=[.4, .4, .4])
                axarr[2].scatter(TH, (seq.B-seq.STIM)/30, s=100, color=color, alpha=.5, lw=1, edgecolors=[.4, .4, .4])
                axarr[3].scatter(np.mean(maxes), np.max(data.s[seq.STIM:seq.E]), s=100, color=color, alpha=.5, lw=1, edgecolors=[.4, .4, .4])

            axarr[5].scatter(seq.STIM, TH, color=color, alpha=.5)

for outcome, ths in outcomes.items():
    if outcome in ('aborted', 'incomplete'): continue
    axarr[4].hist(ths, label=outcome, alpha=.4, density=True)
axarr[4].legend()
axarr[4].set(ylabel='density', xlabel='Mean DFF TH')

axarr[0].set(ylabel='A tag RT', xlabel='Mean DFF TH (per FOV)')
axarr[1].set(ylabel='H tag RT', xlabel='Mean DFF TH (per FOV)')
axarr[2].set(ylabel='B tag RT', xlabel='Mean DFF TH (per FOV)')
_ = axarr[3].set(ylabel='Max escape speed', xlabel='Max DFF during escape (per FOV)')

# %%
# Plt THs difference for escapes and contro runs
thresholds, control_thresholds = [], []


for nsess, sess in enumerate(Sessions.fetch(as_dict=True)):
    print(sess['mouse'], sess['date'])
    data, rois = get_session_data(sess['mouse'], sess['date'], roi_data_type='raw')
    
    # Get which ROIs show escape-related acivity
    active_rois = get_active_rois(rois, sequences, sess)

    tags = get_session_tags(sess['mouse'], sess['date'], 
                        etypes=('visual', 'audio', 'audio_visual'), 
                        ttypes=('A' , 'H', 'B', 'C', 'E'))

    control_tags = get_session_tags(sess['mouse'], sess['date'], 
                    etypes=('control',), 
                    ttypes=('A', 'H', 'B', 'C', 'E'))
    
    # get tags sequences
    sequences = get_tags_sequences(tags)
    control_sequences = get_tags_sequences(control_tags)

    lengths = len([s for s in sequences if seq_type(s)=='complete']),  len(control_sequences)
    print(f'Escape and control number of sequences: {lengths}')


    for tp, sink, seqs in zip(('escape', 'control'), (thresholds, control_thresholds), (sequences, control_sequences)):
        for seq in seqs:
            if tp == 'escape' and seq_type(seq) != 'complete':
                continue

            start = seq.STIM - int(3 * 30)
            if not data['is_rec'][start]: continue

            ths = []
            for roi in active_rois.columns:
                ths.append(np.nanpercentile(active_rois[roi][start:seq.STIM], 30))

            if np.nanmean(ths) < .1:
                raise ValueError
            sink.append(np.nanmean(ths))

# %%
f, ax = plt.subplots(figsize=(12, 12))

ax.hist(thresholds, bins=np.linspace(0, 20, 20), color='blue', alpha=.5, density=True, label='escape')
ax.hist(control_thresholds, bins=np.linspace(0, 20, 20), color='salmon', alpha=.5, density=True, label='control')
ax.legend()
# %%
# %%
from scipy.stats import ttest_ind
ttest_ind(thresholds, control_thresholds, equal_var=False)
# %%
pvals = []
for i in range(10000):
    t = np.random.choice(thresholds, len(thresholds))
    c = np.random.choice(control_thresholds, len(thresholds))
    pvals.append(ttest_ind(t, c).pvalue)

_ = plt.hist(pvals, bins=np.linspace(0, .6, 100))
x = np.mean(pvals)
plt.axvline(x, color='red')
# %%
