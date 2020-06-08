import sys
sys.path.append("./")

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import preprocessing

from fcutils.plotting.colors import desaturate_color
from fcutils.plotting.colors import *
from fcutils.plotting.utils import clean_axes, save_figure, set_figure_subplots_aspect
from fcutils.plotting.quick_plots import quick_scatter, quick_plot
from fcutils.file_io.utils import check_create_folder
from fcutils.maths.utils import derivative

from vgatPAG.database.db_tables import Trackings, Session, Roi, Recording, Mouse, Event, TiffTimes
from vgatPAG.paths import single_roi_models


np.warnings.filterwarnings('ignore')

DEBUG = False

# Get all mice
mice = Mouse.fetch("mouse")

# Get all sessions
sessions = {m:(Session & f"mouse='{m}'").fetch("sess_name") for m in mice}

# Get the recordings for each session
recordings = {m:{s:(Recording & f"sess_name='{s}'" & f"mouse='{m}'").fetch(as_dict=True) for s in sessions[m]} for m in mice}


# Loop over sessions
for mouse in mice:
    for sess in sessions[mouse]:
        # Fetch some data
        recs = Recording().get_sessions_recordings(mouse, sess)
        roi_ids, roi_sigs, nrois = Roi().get_sessions_rois(mouse, sess)
        roi_ids = list(roi_ids.values())[0]


        # Loop over each ROI
        print(f"Mouse {mouse} - session: {sess} - [{len(roi_ids)} rois]")
        for n in tqdm(range(list(nrois.values())[0])):
            if DEBUG and roi_ids[n] != "CRaw_ROI_2": continue

            # Loop over each recording
            calcium_trace = []
            behaviour_traces = dict(
                speed = [],
                acceleration = [],
                shelter_distance = [],
                ang_vel = [],
                # shelter_acceleration = [],
            )
            
            for rec in recs:
                # Get tracking data aligned to calcium recording
                body_tracking, ang_vel, speed, shelter_distance = Trackings().get_recording_tracking_clean(sess_name=sess, rec_name=rec)
                ang_vel = ang_vel[1:] # drop first frame to match the calcium frames
                speed = speed[1:]
                shelter_distance = shelter_distance[1:]


                # Get only time points when Calcium recording is on
                rsig = Roi().get_roi_signal_clean(rec, roi_ids[n])

                signal = (TiffTimes & f"rec_name='{rec}'").fetch1("is_ca_recording")
                rec_on = np.where(signal)[0]
                speed = speed[rec_on]
                ang_vel = ang_vel[rec_on]
                shelter_distance = shelter_distance[rec_on]

                # Add stuff to traces
                calcium_trace.extend(list(rsig))
                behaviour_traces['speed'].extend(list(speed))
                behaviour_traces['acceleration'].extend(list(derivative(speed)))
                behaviour_traces['ang_vel'].extend(list(ang_vel))
                behaviour_traces['shelter_distance'].extend(list(shelter_distance))


            # Keep only times where mouse is out of shelter
            th = 2
            keep_idx = np.where(np.array(behaviour_traces['speed']) > th)[0]

            btraces = {k:np.array(v)[keep_idx] for k,v in behaviour_traces.items()}
            ctrace = np.array(calcium_trace)[keep_idx]

            # Fit robust linear regression model
            exog = pd.DataFrame({k:v for k,v in btraces.items()}).interpolate()
            # exog = exog.drop(columns=['speed'])

            # add interaction terms
            # exog['shelt_speed'] = - derivative(exog['shelter_distance'].values)
            # exog['shelt_accel'] = derivative(exog['shelt_speed'].values)
            # exog['speedxsdist'] = (exog['shelt_speed'] * exog['shelter_distance'])

            # NOrmalize and prepare endog
            exog = pd.DataFrame(preprocessing.scale(exog.values, axis=0), columns=exog.columns, index=exog.index)
            exog = sm.add_constant(exog, prepend=False)
            endog = pd.DataFrame({"signal":ctrace})

            # Fit and save
            model = sm.RLM(endog.reset_index(drop=True), exog.reset_index(drop=True)).fit() 
            model_name = f"{mouse}_{sess}_roi_{roi_ids[n]}_speedth_{th}.pkl"
            model.save(os.path.join(single_roi_models, model_name))

            # if DEBUG: 
            #     print(model.summary())
            #     f, ax = quick_plot(endog['signal'], model.predict(exog))
            #     ax.axhline(np.nanmean(ctrace), color='r')
            #     plt.show()
            #     break


        if DEBUG: break
    if DEBUG: break



