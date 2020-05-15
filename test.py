# %%
from vgatPAG.database.db_tables import *
from vgatPAG.analysis.utils import get_mouse_session_data, get_session_stimuli_frames, get_shelter_threat_trips

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from random import choices

from behaviour.utilities.signals import get_times_signal_high_and_low

from brainrender.colors import colorMap

from fcutils.plotting.plot_elements import plot_shaded_withline
from fcutils.plotting.plot_distributions import plot_kde
from fcutils.plotting.colors import *
from fcutils.plotting.utils import save_figure, clean_axes

# Get all mice
mice = Mouse.fetch("mouse")

# Get all sessions
sessions = {m:(Session & f"mouse='{m}'").fetch("sess_name") for m in mice}

# Get the recordings for each session
recordings = {m:{s:(Recording & f"sess_name='{s}'" & f"mouse='{m}'").fetch(as_dict=True) for s in sessions[m]} for m in mice}

mouse = mice[0]
sess = sessions[mouse][0]
recs = Recording().get_sessions_recordings(mouse, sess)

print(recs)

# Make figure 
f, axarr = plt.subplots(ncols=6, nrows=6, figsize=(18, 9), sharex=True)
f.suptitle('RED: shelter to threat - BLUE: threat to shelter')
axarr = axarr.flatten()

# Get tracking
tracking, ang_vel, speed, shelter_distance, signals, _nrois, is_rec = get_mouse_session_data(mouse, sess, sessions)
speed[speed>np.nanpercentile(speed, 99)] = np.nanpercentile(speed, 99)

stimuli = get_session_stimuli_frames(mouse, sess)

for nroi in np.arange(_nrois):
    trace = pd.DataFrame(dict(
            x = np.int64(tracking['x'].values),
            x_speed = derivative(tracking.x.values),
            y = np.int64(tracking['y'].values),
            s =speed,
            sig = signals[nroi],
            isrec = is_rec,
        )).interpolate()

    # Get the shelter->threat and threat-> shelter trips
    shelter_to_threat, threat_to_shelter = get_shelter_threat_trips(trace, stimuli=stimuli, min_frames_after_stim=10000*30)
    print(threat_to_shelter)