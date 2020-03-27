"""
    Uses deeplabcut to track all the videos in dataset
"""

import sys
sys.path.append("./")

from vgatPAG.paths import mice_folders, dlc_config_file
from vgatPAG.database.tables import *


try:
    import deeplabcut as dlc
except:
    raise ImportError("Could not import deeplabcut, make sure to use an environment with working dlc installation")


try:
    import fcutils
except:
    raise ImportError("Could not import 'fcutils', please install with : 'pip install git+https://github.com/FedeClaudi/fcutils.git --upgrade' ")
from fcutils.file_io.utils import get_subdirs, listdir, get_file_name


try:
    import behaviour
except:
    raise ImportError("Could not import 'behaviour', please install with : pip install git+https://github.com/BrancoLab/Behaviour.git --upgrade' ")
from behaviour.tracking.tracking import prepare_tracking_data, compute_body_segments



bsegments = [ # Body segments used to get stuff like angular velocity, orientation etc
    ['snout', 'neck'],
    ['neck', 'body'],
    ['body', 'tail_base']
]



# ---------------------------------------------------------------------------- #

#                                   TRACKING                                   #
# ---------------------------------------------------------------------------- #
# Loop over each mouse's folder to get the list of files not tracked
videos_to_process = []

recordings = Recording().fetch(as_dict=True)
for rec in recordings:
    video = rec['videofile']
    fld = get_session_folder(**rec)

    dlc_analysed = [f for f in listdir(fld) if 'DLC' in f and f.endswith(".h5") and get_file_name(video) in f]
    if not dlc_analysed:
        videos_to_process.append(os.path.join(fld, video))


# Start tracking
print(f"Found {len(videos_to_process)} videos not tracked:")
print(*videos_to_process, sep="\n")
print("Firing DLC up...")

dlc.analyze_videos(dlc_config_file, 
                videos_to_process, 
                save_as_csv=False,
                dynamic=(True, 0.5, 100))



