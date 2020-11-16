import sys
sys.path.append("./")


from vgatPAG.database.db_tables import *


"""
    Creates a datajoint dabase from the summary .hdf file to facilitate files handling
"""

# ManualBehaviourTags.drop()


# -------------------------- experiment-based tables ------------------------- #

# Mouse.populate()
# Experiment.populate(display_progress=True)
# # CaFPS.populate(display_progress=True)
# Trackings.populate(display_progress=True)


# --------------------------- Session based tables --------------------------- #
# Sessions.populate(display_progress=True)
# Roi.populate(display_progress=True)

ManualBehaviourTags.populate(display_progress=True)
