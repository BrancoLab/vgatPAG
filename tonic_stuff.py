import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from brainrender._colors import map_color

from pyinspect import install_traceback
install_traceback()

from fcutils.plotting.utils import calc_nrows_ncols, clean_axes, save_figure
from fcutils.maths.utils import rolling_mean, derivative
from vgatPAG.database.db_tables import ManualBehaviourTags, Roi, Sessions

from Analysis import get_session_data, get_session_tags, get_tags_sequences, get_active_rois

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


# %%
f, axarr = plt.subplots(ncols=2, nrows = 2, figsize=(12, 12))
S = len(Sessions.fetch())
for nsess, sess in enumerate(Sessions.fetch(as_dict=True)):
    print(sess['mouse'], sess['date'])
    color = map_color(nsess, 'viridis', 0, S)


    data, rois = get_session_data(sess['mouse'], sess['date'], roi_data_type='raw')
    if data is None: 
        continue

    tags = get_session_tags(sess['mouse'], sess['date'], 
                        etypes=('visual', 'audio', 'audio_visual'), 
                        ttypes=('H', 'B', 'C', 'E'))
    
    # get tags sequences
    sequences = get_tags_sequences(tags)
    colors = [map_color(n, 'viridis', 0, len(sequences)) for n in range(len(sequences))]

    # Get which ROIs show escape-related acivity
    active_rois = get_active_rois(rois, sequences, sess)

    # iterate sequences
    prev_stim = 0
    thresholds = []
    for seq in sequences:
        if seq.E is None or seq.C is None or seq.B is None or seq.H is None: 
            continue
        if seq.STIM - prev_stim < 60*30:
            continue
        else:
            pre_stim = seq.STIM

        start = seq.STIM - 5 * 30
        if data.is_rec[start] != 0:
            # Compute mean DFF threshold
            ths = []
            for roi in active_rois.columns:
                th = np.nanpercentile(active_rois[roi][start:seq.STIM], .3)
                ths.append(th)
            TH = np.mean(ths)
            
            color = [.4 , .4, .4]
            axarr[0, 1].scatter(TH, (seq.H-seq.STIM)/30, s=100, color=color, alpha=.5, lw=1, edgecolors=[.4, .4, .4])
            axarr[1, 0].scatter(TH, (seq.B-seq.STIM)/30, s=100, color=color, alpha=.5, lw=1, edgecolors=[.4, .4, .4])
            axarr[1, 1].scatter(TH, np.max(data.s[seq.STIM:seq.E]), s=100, color=color, alpha=.5, lw=1, edgecolors=[.4, .4, .4])

            thresholds.append(TH)

    # axarr[0, 0].hist(thresholds, color=color, alpha=.5, bins = np.linspace(0, 18, 9))
    # break

axarr[0, 0].set(ylabel='count', xlabel='Mean DFF TH (per FOV)')
axarr[0, 1].set(ylabel='H tag RT', xlabel='Mean DFF TH (per FOV)')
axarr[1, 0].set(ylabel='B tag RT', xlabel='Mean DFF TH (per FOV)')
_ = axarr[1, 1].set(ylabel='Max escape speed', xlabel='Mean DFF TH (per FOV)')


# %%