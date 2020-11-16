import sys
sys.path.append("./")


from vgatPAG.database.db_tables import *


"""
    Creates a datajoint dabase from the summary .hdf file to facilitate files handling
"""

# Mouse.drop()


# -------------------------- experiment-based tables ------------------------- #

# Mouse.populate()
# Experiment.populate(display_progress=True)
# # CaFPS.populate(display_progress=True)
# Trackings.populate(display_progress=True)


# --------------------------- Session based tables --------------------------- #
Sessions.populate(display_progress=True)
Roi.populate(display_progress=True)




# TODO populate tracking
# TODO populate manual tags
# TODO check tags - dff alignments
# TODO plot traces aligned to tags
# TODO enjoy

# ----------------------------- Manual behav tags ---------------------------- #
# print('Populating behaviour tags')
# ManualBehaviourTags.populate(display_progress=True)

# # --------------------------------- Tracking --------------------------------- #
# print("Populating Tracking")


# # ---------------------------------- Pop ROI --------------------------------- #
# print("Popualte ROIs")
# Roi.populate(display_progress=True)
# RoiDFF.populate(display_progress=True)

# # # ---------------------------- Pop manually curate --------------------------- #
# print("Populate manually curated data")
# ManualTrials.populate()
# ManualROIs.populate(display_progress=True)


# print(Times())