# %%
import sys
sys.path.append("./")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from fcutils.plotting.colors import colorMap, desaturate_color
from fcutils.plotting.colors import *
from fcutils.plotting.utils import create_figure, clean_axes, save_figure
from fcutils.objects import sort_list_of_strings

from vgatPAG.database.db_tables import ManualROIs as ROIs
from vgatPAG.paths import output_fld
from vgatPAG.variables import miniscope_fps, tc_colors

fps = miniscope_fps
sec_pre = 20
# sec_post = 50





# %%
# ---------------------------------------------------------------------------- #
#                         PLOT ALL TRACES FOR A SESSION                        #
# ---------------------------------------------------------------------------- #
rois = ROIs()
mice = rois.get_mice()

for mouse in mice:
    sessions = rois.get_mouse_sessions(mouse)
    for sess in sessions:
        trials = rois.get_session_trials(mouse, sess)
        sess_rois = list(set(trials.roi_id.values))
        sort_list_of_strings(sess_rois)

        f, axarr = plt.subplots(ncols=len(sess_rois), nrows=len(tc_colors.keys()),
                    sharex=True, sharey=True, figsize=(30, 20))
        f.suptitle(f"{mouse} - {sess}")
        for t, (tc, color) in enumerate(tc_colors.items()):
            for r, roi in enumerate(sess_rois):
                traces = list(trials.loc[(trials.trial_class == tc)&(trials.roi_id== roi)].signal.values)
                for trace in traces:
                    axarr[t, r].plot(trace, color=color, lw=1.5, alpha=.5)

                axarr[t, r].axvline(sec_pre*fps, lw=3, color='k', ls="--", alpha=.3)
                axarr[t, r].set(ylim=[-100, 300])
                if r == 0:
                    axarr[t, r].set(ylabel=tc)
                
        
                if t == 0:
                    axarr[t, r].set(title=f"{roi}")

    
        clean_axes(f)
        f.tight_layout()
        save_figure(f, os.path.join(output_fld, f"{mouse}_{sess}_alltrials"))
        



# %%

# ---------------------------------------------------------------------------- #
#                  PLOT TRACES FOR EACH TRIAL IN A TRIAL CLASS                 #
# ---------------------------------------------------------------------------- #

# ------------------------ Get data for a trial class ------------------------ #
trials_class = "Loom_Escape"
rois = ROIs()
trials, mice, sessions, trials_by_mouse, trials_by_session = rois.get_trials_in_class(trials_class)


# %%
# ---------------------- Plot traces by session by mouse by trial --------------------- #


cmap = "Reds"

for mouse in mice:
    for sess in sessions[mouse]:
        trs = trials_by_session[sess]
        sess_rois = list(set(trs.roi_id.values))
        sess_trs = list(set(trs.trial_name.values))


        f, axarr = plt.subplots(ncols=len(sess_rois), figsize=(30, 6), sharex=True, sharey=True)
        for roi, ax in zip(sorted(sess_rois), axarr):
            roi_traces = list(trs.loc[trs.roi_id == roi].signal.values)

            # Plot individual traces
            for trace in roi_traces:
                ax.plot(trace, lw=1.5, alpha=.5, color=salmon)


            # Plot mean
            ax.plot(np.mean(roi_traces, 0), lw=3, color=[.2, .2, .2])


            # Decorate figure
            ax.axvline(sec_pre*fps, lw=3, color='k', ls="--", alpha=.5)
            ax.set(title=f"{roi}")
        f.suptitle(f"{mouse} - {sess}")
        break
    break

        




# %%
