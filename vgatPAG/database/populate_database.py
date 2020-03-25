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



# ---------------------------- Pop manually curate --------------------------- #
print("Populate manually curated data")
ManualTrials.populate()
ManulROIs.populate(display_progress=True)

