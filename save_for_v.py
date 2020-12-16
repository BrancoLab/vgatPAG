from pathlib import Path
import pandas as pd
import numpy as np

from fcutils.maths.utils import rolling_mean

from vgatPAG.database.db_tables import ManualBehaviourTags, Roi, Sessions
from Analysis import get_session_data, get_session_tags, get_tags_sequences, speed_color, shelt_dist_color, get_active_rois


n_frames_pre = 5 * 30
n_frames_post = 5 * 30

fld = Path('D:\\Dropbox (UCL)\\Project_vgatPAG\\analysis\\doric\\Fede\\ForV')

def dff(sig):
    th = np.nanpercentile(sig[:n_frames_pre], 30)
    return rolling_mean((sig - th)/th, 3)

for sess in Sessions.fetch(as_dict=True):
    print(sess['mouse'], sess['date'])

    data, rois = get_session_data(sess['mouse'], sess['date'], roi_data_type='raw')
    if data is None: continue
    rois[data.is_rec==0] = np.nan

    tags = get_session_tags(sess['mouse'], sess['date'], 
                        etypes=('visual', 'audio', 'audio_visual'), 
                        ttypes=('H', 'B', 'C', 'E'))

    # get tags sequences
    sequences = get_tags_sequences(tags)

        # Loop over sequences
    prev_stim = 0
    for sn, seq in enumerate(sequences):     
        # check we didn't have a stim too  close
        if seq.STIM - prev_stim < 60*30:
            continue
        else:
            pre_stim = seq.STIM

        # Check for aborted escapes
        if seq.E is None or seq.C is None or seq.B is None or seq.H is None: continue
        
        start = seq.STIM - n_frames_pre
        end = seq.E + n_frames_post
        if data.is_rec[start] == 0: continue  # stim when not recording

        # Get each ROIs DFF and save to file
        trial_data = {}
        for roi in rois.columns:
            trial_data[roi] = dff(rois[roi][start:end])

        pd.DataFrame(trial_data).to_hdf(fld / f'{sess["mouse"]}_{sess["date"]}_{seq.STIM}', key='hdf')

