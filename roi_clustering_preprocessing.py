# %%
import os
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from pathlib import Path
from tqdm import tqdm

from numba import jit
from affinewarp import ShiftWarping, PiecewiseWarping

from scipy.stats import zscore

from fcutils.maths.filtering import smooth_hanning

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

n_sec_pre = 4 # rel escape onset
n_sec_post = 6 # # rel escape onset
n_frames_pre = n_sec_pre * fps
n_frames_post = n_sec_post * fps

n_sec_baseline = 15 # n seconds before stimulus to use for DF/F calculation

n_SD_th = 1 # trials are kept is signal is above or beyond this n of SDs from mean


DFF = True # if true DF/F signal
FILTER = True # if true filter signal
ZSCORE = True


# %%

# ---------------------------------------------------------------------------- #
#                                 CACHE SIGNALS                                #
# ---------------------------------------------------------------------------- #


# TODO exclude trials in which response <= noise
# TODO add time warping before coorelation


def process_sig(sig, baseline=None,  dff=False, filt=False, zsc=False):
    if filt: # median filter
        sig = smooth_hanning(sig, window_len=21)
    if dff: # dff
        sig =  (sig - np.mean(baseline))  / np.mean(baseline)

    if zsc:
        sig = zscore(sig)
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
        baseline = [], 
        x = [], 
        y = [],
        s = [],
        above_noise = [],
        full_escape = [], 
        at_shelt_frame = [],
    )


    tag_type = 'VideoTag_B'
    event_type =  ['Loom_Escape', 'US_Escape', 'LoomUS_Escape'] #  


    
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
            sig = process_sig(sig[is_rec == 1], dff=False, filt=FILTER, zsc=False)
            baselines.append((np.mean(sig) - n_SD_th * np.std(sig), np.mean(sig) + n_SD_th * np.std(sig)))


        for count, (i, tag) in tqdm(enumerate(tags.iterrows())):
            start = tag.session_frame - n_frames_pre
            end = tag.session_frame + n_frames_post

            nxt_at_shelt = get_next_tag(tag.session_frame, at_shelt_tags)
            if nxt_at_shelt is None: 
                full_escape = False
                at_shelt_frame = None
            else:
                full_escape = True
                at_shelt_frame = nxt_at_shelt - start



            # CACHE
            for roin, sig in enumerate(signals):
                if np.any(is_rec[tag.session_stim_frame - fps * n_sec_baseline:end] == 0): # recording was off
                    print('no rec')
                    continue

                    
                above_noise = np.where((sig[start:end] < baselines[roin][0]) | (sig[start:end] > baselines[roin][1]))
                if not np.any(above_noise) :
                    # 'Signal not out of noise
                    above_noise = False
                else:
                    above_noise = True


                # Get signal and baseline
                baseline = process_sig(sig[tag.session_stim_frame - fps * n_sec_baseline: tag.session_stim_frame], filt=FILTER, zsc=False)
                sig = sig[start:end]
                
                sig = process_sig(sig,  baseline, dff=DFF, filt=FILTER, zsc=ZSCORE)


                cache['mouse'].append(mouse)
                cache['session'].append(sess)
                cache['session_frame'].append(tag.session_frame)
                cache['tag_type'].append(tag_type)
                cache['n_frames_pre'].append(n_frames_pre)
                cache['n_frames_post'].append(n_frames_post)
                cache['roi_n'].append(roin)
                cache['above_noise'].append(above_noise)
                cache['full_escape'].append(full_escape)
                cache['at_shelt_frame'].append(at_shelt_frame)

                cache['signal'].append(sig)
                cache['baseline'].append(baseline)

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
# Plot some traces

sigs = np.vstack([r.signal for i,r in cache.iterrows()]).T

_  = plt.plot(sigs)


# %%

# ---------------------------------------------------------------------------- #
#                            COMPUTE CROSS CORR MTX                            #
# ---------------------------------------------------------------------------- #
if COMPUTE:
    cache = cache.loc[cache.above_noise == True]
    sigs = [t.signal for i,t in cache.iterrows()]

    # corr = make_corr_mtx(sigs)
    corr = np.corrcoef(sigs)
    
    np.save(os.path.join(fld, 'cached_corr_mtx.npy'), corr)
    print('finished')
else:
    corr = np.load(os.path.join(fld, 'cached_corr_mtx.npy'))

plt.imshow(corr)

# %%

# %%

# %%
