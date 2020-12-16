# %%
import matplotlib.pyplot as plt
import numpy as np
from rich.progress import track
from scipy.stats import zscore
from random import choices

from fcutils.plotting.utils import clean_axes, save_figure
from fcutils.maths.utils import rolling_mean

from vgatPAG.database.db_tables import ManualBehaviourTags, Roi, Sessions
from Analysis import get_session_data, get_session_tags, get_tags_sequences, get_active_rois, seq_type

# %%
def dff(sig):
    th = np.nanpercentile(sig[: - N_SEC_PRE * 30], 30)
    return rolling_mean((sig - th)/th, 3)

def valid(seq):
    # check valid sequence
    if seq.STIM - prev_stim < 60*30:
        return False
    else:
        if data.is_rec[seq.STIM - 5 * 30] == 0:
            return False
        if seq_type(seq) != 'complete':
            return False
        return seq.STIM

def before(n):
    return n - N_SEC_PRE*30
def after(n):
    return n + N_SEC*30
    
def split_traces(traces, roin):
    
    n = len(traces['STIM'])

    one_idxs = choices(np.arange(n), k=int(n*.6))

    if roin == 0:
        print(f'Cross validation set has {len(one_idxs)} trials and control set has {n - len(one_idxs)} trials')

    one = {k:[vv for i,vv in enumerate(v) if i in one_idxs] for k,v in traces.items()}
    two = {k:[vv for i,vv in enumerate(v) if i not in one_idxs] for k,v in traces.items()}

    return one, two


# %%
'''
    Get the mean response of each ROI aligned to a tag
'''
N_SEC_PRE = 2
N_SEC = 6
responses = [{k:[] for k in ('STIM', 'A', 'H', 'B', 'C', 'E')} for i in range(2)]
control_responses = [{k:[] for k in ('STIM', 'A', 'H', 'B', 'C', 'E')} for i in range(2)]
for sess in track(Sessions.fetch(as_dict=True), total=len(Sessions())):
    data, rois = get_session_data(sess['mouse'], sess['date'], roi_data_type='raw')
    if data is None: continue
    rois[data.is_rec==0] = np.nan

    tags = get_session_tags(sess['mouse'], sess['date'], 
                        etypes=('visual', 'audio', 'audio_visual'), 
                        ttypes=('A', 'H', 'B', 'C', 'E'))

    control_tags = get_session_tags(sess['mouse'], sess['date'], 
                        etypes=('control',), 
                        ttypes=('A', 'H', 'B', 'C', 'E'))

    for ttype, tgs, store in zip(('escape', 'control'), (tags, control_tags), (responses, control_responses)):
        # get tags sequences
        sequences = get_tags_sequences(tgs)
        if len(set([s.STIM for s in sequences])) <3: continue

        # get active ROIs
        active_rois = get_active_rois(rois, sequences, sess)

        # Loop over ROIs
        for roin, roi in enumerate(active_rois.columns):
            roi_traces = {k:[] for k in store[0].keys()}
            # iterate sequences
            prev_stim = 0
            for seq in sequences:
                if not valid(seq) and ttype != 'control':
                    continue
                else:
                    prev_stim = valid(seq)

                # dff
                start = seq.STIM - N_SEC_PRE * 30

                if ttype == 'escape':
                    end = seq.E + N_SEC * 30
                else:
                    end = seq.STIM + N_SEC * 30
                sig = dff(active_rois[roi][start:end])

                rel_seq = {k:n-seq.STIM + N_SEC_PRE * 30 if n is not None else n for k,n in seq._asdict().items()}

                # Get mean of tag aligned traces for each roi
                for k, frame in rel_seq.items():
                    if k  == 'D' or frame is None:
                        continue
                    if ttype == 'control' and k == 'B':
                        continue

                    if frame:
                        if len(sig[before(frame):after(frame)]) != (N_SEC_PRE + N_SEC) * 30:
                            print('Skipping trial of incorrect length')
                            continue
                        roi_traces[k].append(sig[before(frame):after(frame)])
                
            # Split traces for cross validation
            for n, traces in enumerate(split_traces(roi_traces, roin)):
                for k,v in traces.items():
                    if k  == 'D':
                        continue
                    if not v:
                        continue
                    store[n][k].append(np.nanmean(np.vstack(v), 0))

    #     break
    # break

# %%
# Make figure
f, axarr = plt.subplots(ncols=2, nrows=6, figsize=(16, 20), sharex=True)

vmin = np.min(zscore(np.hstack(responses[0].values()), 1))
vmax = np.max(zscore(np.hstack(responses[0].values()), 1))

names = dict(
    STIM = 'simulus',
    A = 'startle',
    H = 'escape start',
    B = 'run start',
    C = 'shelter enter',
    E = 'escape end'
)

for axes, title, data in zip((axarr[:, 0], [axarr[3, 1],]), ('escape', 'control'), (responses, control_responses)):
    for ax, (name, values) in zip(axes, data[0].items()):
        if not values:
            continue
        
        values = np.array(values)
        at_max = np.argmin(values, 1)
        idxs = np.argsort(at_max)

        validated_values = np.array(data[0][name])[idxs, :]
        ax.imshow(zscore(validated_values, 1), cmap='hot' , vmin=vmin, vmax=6)

        name = names[name]
        ax.set(
            title=name if name != 'STIM' else f'{title} | {name}',
            xticks = [0, N_SEC_PRE*30, (N_SEC_PRE+N_SEC)*30],
            xticklabels = [-N_SEC_PRE, 0, N_SEC],
            xlabel='time (s)' if name == 'E' else None,
            ylabel='ROI #',
        )
        ax.axvline(N_SEC_PRE*30, lw=2, color='w', alpha=.8)

for ax in axarr[:, 1]:
    ax.axis('off')
f.tight_layout()
    # %%
