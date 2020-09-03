# %%
import os
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import medfilt
from pathlib import Path
from tqdm import tqdm
from scipy.stats.stats import pearsonr   

from numba import jit
from affinewarp import ShiftWarping, PiecewiseWarping


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
    get_next_tag,
    get_last_tag,
)


""""
    Cache tracking and calcium traces for each ROI/trial as a .h5 file
    with a pandas dataframe. 

    Then use the cached data for creating a correlation matrix
"""
# %%

# ---------------------------------------------------------------------------- #
#                                    PARAMS                                    #
# ---------------------------------------------------------------------------- #
fld = Path('D:\\Dropbox (UCL - SWC)\\Project_vgatPAG\\analysis\\doric\\Fede\\plots\\ManualTagsAligned')
CACHE_SIGNALS = True # if False it simply loads a previous cache
COMPUTE = True # if true it computes a new corr.mtx otherwise it loads one


fps = 30
n_sec_pre = 4
n_sec_post = 4
n_frames_pre = n_sec_pre * fps
n_frames_post = n_sec_post * fps


# %%

# ---------------------------------------------------------------------------- #
#                                 CACHE SIGNALS                                #
# ---------------------------------------------------------------------------- #


# TODO exclude trials in which response <= noise
# TODO add time warping before coorelation


def process_sig(sig, start, end, n_sec_post, norm=False, filter=True):
    if filter: # median filter
        sig = medfilt(sig, kernel_size=5)
    if norm: # BASELINE with baseline
        baseline = np.mean(sig[start: start + n_frames_pre - 1])
        sig =  sig - baseline
    return sig




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
        x = [], 
        y = [],
        s = [],
        above_noise = [],
    )


    tag_type = 'VideoTag_B'
    event_type =  ['Loom_Escape', 'US_Escape', 'LoomUS_Escape'] #  

    BASELINE = True # if true basleine the sginal
    FILTER = True # if true data are median filtered to remove artefact

    
    for mouse, sess, sessname in mouse_sessions:
        # if sessname not in include_sessions: continue
        print(f'Processing {sessname}\n')

        # Get data
        tracking, ang_vel, speed, shelter_distance, signals, nrois, is_rec = get_mouse_session_data(mouse, sess, sessions)
        tags = get_tags_by(mouse=mouse, sess_name=sess, event_type=event_type, tag_type=tag_type)
        
        at_shelt_tags = get_tags_by(mouse=mouse, sess_name=sess, event_type=event_type, tag_type='VideoTag_E')

        # For each signal, get the mean and baseline 
        baselines = []
        for sig in signals:
            sig = sig[is_rec == 1]
            baselines.append((np.mean(sig) - 1 * np.std(sig), np.mean(sig) + 1 * np.std(sig)))


        for count, (i, tag) in tqdm(enumerate(tags.iterrows())):
            start = tag.session_frame - n_frames_pre
            end = tag.session_frame + n_frames_post

            nxt_at_shelt = get_next_tag(tag.session_frame, at_shelt_tags)
            if nxt_at_shelt is None: 
                print('no shelt')
                continue

            if np.sum(is_rec[start:end]) == 0: # recording was off
                continue

            # CACHE
            for roin, sig in enumerate(signals):
                if np.std(sig[start:end]) == 0: # recording was off
                    raise ValueError
                
                sig = process_sig(sig, start, end, n_sec_pre, norm=BASELINE, filter=FILTER)
                sig = sig[start:end]

                above_noise = np.where((sig < baselines[roin][0]) | (sig > baselines[roin][1]))
                if not np.any(above_noise) :
                    # 'Signal not out of noise
                    above_noise = False
                else:
                    above_noise = True

                cache['mouse'].append(mouse)
                cache['session'].append(sess)
                cache['session_frame'].append(tag.session_frame)
                cache['tag_type'].append(tag_type)
                cache['n_frames_pre'].append(n_frames_pre)
                cache['n_frames_post'].append(n_frames_post)
                cache['roi_n'].append(roin)
                cache['above_noise'].append(above_noise)

                cache['signal'].append(sig)

                cache['x'].append(tracking.x[start:end].values)
                cache['y'].append(tracking.y[start:end].values)
                cache['s'].append(speed[start:end])

    cache = pd.DataFrame(cache)
    cache.to_hdf(os.path.join(fld, 'cached_traces.h5'), key='hdf')
else:
    cache = pd.read_hdf(os.path.join(fld, 'cached_traces.h5'), key='hdf')

print(f'\n\n\n{len(cache)} traces in the cache')
cache.head()

# %%

# ---------------------------------------------------------------------------- #
#                            COMPUTE CROSS CORR MTX                            #
# ---------------------------------------------------------------------------- #

# @jit(nopython=True)
def make_corr_mtx(sigs):

    n_sigs = len(sigs)

    corr = np.zeros((n_sigs, n_sigs))

    done = []
    for i in tqdm(range(n_sigs)):
        print(i)
        for j in range(n_sigs):
            if (i, j) in done or (j, i) in done: 
                continue
            else:
                done.append((i, j))

            if i == j:
                corr[i, j] = 1.
            else:
                _corr = pearsonr(sigs[i, :], sigs[j, :])[0]
                # _corr = np.corrcoef(sigs[i, :], sigs[j, :])[0, 1]
                corr[i, j] = _corr
                corr[j, i] = _corr
    
    return corr





if COMPUTE:
    cache = cache.loc[cache.above_noise == True]
    sigs = np.vstack([t.signal for i,t in cache.iterrows()])

    corr = make_corr_mtx(sigs)
    # corr = np.corrcoef(sigs)
    
    np.save(os.path.join(fld, 'cached_corr_mtx.npy'), corr)
    print('finished')
else:
    corr = np.load(os.path.join(fld, 'cached_corr_mtx.npy'))

    

# %%
