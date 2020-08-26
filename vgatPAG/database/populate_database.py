import sys
sys.path.append("./")

from fcutils.file_io.io import open_hdf

from vgatPAG.paths import summary_file
from vgatPAG.database.db_tables import *
from vgatPAG.database.dj_config import manual_insert_skip_duplicate


"""
    Creates a datajoint dabase from the summary .hdf file to facilitate files handling
"""
# ManualBehaviourTags.drop()



# # --------------------------- Pop Mouse and Session -------------------------- #
# print("Populating sessions")
# Mouse.populate()
# Session.populate()
# Recording.populate()


# # -------------------------------- Pop Stimuli ------------------------------- #
# print("Stimuli")
# VisualStimuli.populate(display_progress=True)
# AudioStimuli.populate(display_progress=True)

# # ----------------------- Pop AI, metadata and stimuli ----------------------- #
# print("Populating Tiff Times")
# TiffTimes.populate(display_progress=True)


# # ----------------------------- Manual behav tags ---------------------------- #
# print('Populating behaviour tags')
ManualBehaviourTags.populate(display_progress=True)

# # --------------------------------- Tracking --------------------------------- #
# print("Populating Tracking")
# Trackings.populate(display_progress=True)


# # ---------------------------------- Pop ROI --------------------------------- #
# print("Popualte ROIs")
# Roi.populate(display_progress=True)


# # -------------------------------- Pop events -------------------------------- #
# print("Popualte Events")
# Event.populate(display_progress=True)

# # ---------------------------- Pop manually curate --------------------------- #
# print("Populate manually curated data")
# ManualTrials.populate()
# ManualROIs.populate(display_progress=True)


