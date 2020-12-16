from vgatPAG.database.db_tables import ManualBehaviourTags, Roi, Sessions
import pandas as pd
from collections import namedtuple

def seq_type(seq):
    if seq.D is None:
        if seq.A is None or seq.H is None or seq.B is None or seq.E is None or seq.C is None:
            return 'incomplete'
        return 'complete'
    elif seq.B is None or seq.H is None:
        return 'failed'
    else:
        return 'aborted'

def get_tags_sequences(tags):
    sequence = namedtuple('sequence', 'STIM, A, H, B, C, E, D')
    sequences = []
    
    # get stimuli
    stims = tags.session_stim_frame.unique()

    for n, stim in enumerate(stims):
        seq = [stim]
        tgs = tags.loc[(tags.session_stim_frame == stim)&(tags.session_frame>=stim)]
        if n < len(stims)-1:
            tgs = tgs.loc[tgs.session_frame < stims[n+1]]

        for ttype in ('A', 'H','B','C','E', 'D'):
            nxt = tgs.loc[(tgs.tag_type==f'VideoTag_{ttype}')]

            if nxt.empty:
                seq.append(None)
            else:
                if nxt.session_frame.values[0] - stim > 15*30:
                    seq.append(None)
                else:
                    seq.append(nxt.session_frame.values[0])
        sequences.append(sequence(*seq))
    return sequences

# Get session tags
def get_session_tags(mouse, date, etypes=None, ttypes=None):
    info = dict(mouse=mouse, date=date)
    tags = pd.DataFrame((ManualBehaviourTags * ManualBehaviourTags.Tags & info))

    if etypes:
        tags = tags.loc[tags.event_type.isin(list(etypes))]

    if ttypes:
        tgs = [f'VideoTag_{t}' for t in ttypes]
        tags = tags.loc[tags.tag_type.isin(tgs)]

    return tags


# Get session data
def get_session_data(mouse, date, roi_data_type='dff'):
    '''
        Returns all tracking data and ROIs data for a session, 
        as a dataframe
    '''
    # Get tracking data
    info = dict(mouse=mouse, date=date)
    tracking = pd.DataFrame((Sessions * Sessions.Tracking & info).fetch(as_dict=True))
    rois = pd.DataFrame((Sessions * Roi & info).fetch(as_dict=True))

    N = len(rois.iloc[0].dff)-1
    is_rec = tracking.is_ca_rec.values[0]
    data = dict(
        x = tracking.x.values[0][:N],
        y = tracking.y.values[0][:N],
        s = tracking.s.values[0][:N],
        is_rec = tracking.is_ca_rec.values[0][:N],
    )

    rois_data = {}
    for i, roi in rois.iterrows():
        rois_data[roi['id']] = roi[roi_data_type][:-1]
        
    try:
        return pd.DataFrame(data), pd.DataFrame(rois_data)
    except  Exception:
        print(f'Failed to create dfs, lengths: {[len(v) for v in data.values()]} and {[len(v) for v in rois_data.values()]}')
        return None, None