import sys
sys.path.append("./")

from fcutils.file_io.io import open_hdf

from vgatPAG.paths import summary_file
from vgatPAG.database.tables import *
from vgatPAG.database.dj_config import manual_insert_skip_duplicate


"""
    Creates a datajoint dabase from the summary .hdf file to facilitate files handling
"""
# Mouse.drop()

# TODO deal with multiple recordings within the same session
# TODO load raw data, tracking, stimuli etc...

# --------------------------- Pop Mouse and Session -------------------------- #
print("Populating sessions")
Mouse.populate()
Session.populate()
Recording.populate()


# ----------------------- Pop AI, metadata and stimuli ----------------------- #
print("Populating Tiff Times")
TiffTimes.populate()

# ---------------------------------- Pop ROI --------------------------------- #
print("Popualte ROIs")
Roi.populate()




# ---------------------------------------------------------------------------- #
#                                  POP TRIALS                                  #
# ---------------------------------------------------------------------------- #
# ----------------------------- Load summary data ---------------------------- #
f, keys, subkeys, allkeys = open_hdf(summary_file)


# # -------------------------- Populate trial classes -------------------------- #
# for tc in subkeys['all']:
#     manual_insert_skip_duplicate(TrialClass, {'trial_class': tc})

# # ----------------------------- Populate Sessions ---------------------------- #
# sessions = {tc:[] for tc in subkeys['all']}
# for tc in subkeys['all']:
#     tc_sessions = list(dict(f['all'][tc]).keys())
#     sessions[tc].extend(tc_sessions)
#     for session in tc_sessions:
#         manual_insert_skip_duplicate(Session, {'trial_class': tc, 'session': session})

# # ------------------------------ Populate Trials ----------------------------- #
# for tc, ss in sessions.items():
#     for s in ss:
#         trials = list(dict(f['all'][tc][s]).keys())
#         for trial in trials:
#             manual_insert_skip_duplicate(Trial, 
#                                         {
#                                         'trial_class': tc, 
#                                         'session': s,
#                                         'trial_name':trial,
#                                         'frame':int(trial.split("_")[-1])
#                                         })

# # ------------------------------- Populate ROIs ------------------------------ #
# Roi.populate(display_progress=True)

