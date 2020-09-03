import sys
sys.path.append('./')
import numpy as np
import pandas as pd
from vgatPAG.database.db_tables import ManualBehaviourTags

manual_tags = pd.DataFrame((ManualBehaviourTags.Tags).fetch())

def get_tags_by(**kwargs):
    tags = manual_tags.copy()
    for key, value in kwargs.items():
        if isinstance(value, list):
            tags = tags.loc[tags[key].isin(value)]
        else:
            tags = tags.loc[tags[key] == value]

    return tags





def get_next_tag(frame, tags, max_delay = 1000):
    """
        Selects the first tag to happend after a given frame
        as long as not too long a delay happend
    """
    nxt = tags.loc[(tags.session_frame - frame > 0)&(tags.session_frame - frame < max_delay)]
    if not len(nxt): return None
    else: 
        return nxt.session_frame.values[0]

def get_last_tag(frame, tags, max_delay=500):
    nxt = tags.loc[(tags.session_frame - frame < 0)&(np.abs(tags.session_frame - frame) < max_delay)]
    if not len(nxt): 
        return None
    else: 
        return nxt.session_frame.values[-1]