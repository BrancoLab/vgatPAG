from vgatPAG.database.db_tables import ManualBehaviourTags, Roi, Sessions
import pandas as pd

from myterial import  light_blue, amber, light_green, salmon

# define someuseful colors
shelt_dist_color = light_blue
speed_color = amber
ang_vel_colr = light_green
signal_color = salmon

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

    N = len(rois.iloc[0].dff)
    data = dict(
        x = tracking.x.values[0][:N],
        y = tracking.y.values[0][:N],
        s = tracking.s.values[0][:N],
        is_rec = tracking.is_ca_rec.values[0],
    )

    rois_data = {}
    for i, roi in rois.iterrows():
        rois_data[roi['id']] = roi[roi_data_type]
        
    try:
        return pd.DataFrame(data), pd.DataFrame(rois_data)
    except  Exception:
        print(f'Failed to create dfs, lengths: {[len(v) for v in data.values()]} and {[len(v) for v in rois_data.values()]}')
        return None, None