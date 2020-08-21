import sys
sys.path.append('./')

import pandas as pd
from vgatPAG.database.db_tables import ManualBehaviourTags

manual_tags = pd.DataFrame((ManualBehaviourTags.Tags).fetch())

def get_tags_by(**kwargs):
    tags = manual_tags.copy()
    for key, value in kwargs.items():
        tags = tags.loc[tags[key] == value]

    return tags