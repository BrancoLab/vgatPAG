import datajoint as dj
import pandas as pd
import datetime
import numpy as np
from fcutils.file_io.io import open_hdf

from vgatPAG.paths import summary_file
from vgatPAG.database.dj_config import start_connection, dbname, manual_insert_skip_duplicate
schema = start_connection()

def get_trial_content(trial_class, session, trial_name):
    f, keys, subkeys, allkeys = open_hdf(summary_file)
    return dict(f['all'][trial_class][session][trial_name])

@schema
class TrialClass(dj.Manual):
    definition = """
        trial_class: varchar(64)
    """


@schema
class Session(dj.Manual):
    definition = """
        -> TrialClass
        session: varchar(128)
    """

@schema
class Trial(dj.Manual):
    definition = """
        -> Session
        trial_name: varchar(128)
        ---
        frame: int
    """

@schema
class Roi(dj.Imported):
    definition = """
        -> Trial
        name: varchar(128)
        ---
        signal: longblob
    """

    def _make_tuples(self, key):
        trial_content = get_trial_content(key['trial_class'], key['session'], key['trial_name'])
        rois = [k for k in trial_content.keys() if 'Raw_ROI' in k]

        for roi in rois:
            rkey = key.copy()
            rkey['name'] = roi
            rkey['signal'] = trial_content[roi][()]
            
            manual_insert_skip_duplicate(self, rkey)

