"""
    Uses deeplabcut to track all the videos in dataset
"""

import sys
sys.path.append("./")

from vgatPAG.paths import mice_folders, dlc_config_file

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
for fold_n, folder in enumerate(mice_folders):
    print(f"Processing {folder} - {fold_n + 1} of {len(mice_folders)}.")

    # Get subfolders with daily recordings
    subfolds = get_subdirs(folder)

    # Loop over subfolders
    for subfold in subfolds:
        # Get videos
        videos = [f for f in listdir(subfold) if f.endswith('.mp4') or f.endswith('.avi')]
        videos = [f for f in videos if 'dlc' not in f.lower()] # just in case

        # Get DLC tracking files if any
        tracking_files = [get_file_name(f) for f in listdir(subfold) if f.endswith('.h5') and 'dlc' in f.lower()]

        # Get videos not tracked
        not_tracked = [f for f in videos if get_file_name(f) not in tracking_files]
        videos_to_process.extend(not_tracked)

# Start tracking
print(f"Found {len(videos_to_process)} videos not tracked. Firing dlc up...")
dlc.analyze_videos(dlc_config_file, 
                videos_to_process, 
                save_as_csv=False,
                dynamic=(True, 0.5, 100))



# ---------------------------------------------------------------------------- #
#                                POST PROCESSING                               #
# ---------------------------------------------------------------------------- #
# Get all the tracking outputs from dlc and process them with the behaviour package
files_to_process =  []
for fold_n, folder in enumerate(mice_folders):
    # Get subfolders with daily recordings
    subfolds = get_subdirs(folder)

    # Loop over subfolders
    for subfold in subfolds:
        files = [f for f in listdir(subfold) if f.endswith('.h5') and 'dlc' in f.lower()]
        files_to_process.extend(files)

print(f"Processing {len(files_to_process)} files")
for tracking_file in tqdm(files_to_process):
    bp_tracking = prepare_tracking_data(tracking_file, 
                                    median_filter=True,
                                    fisheye=False, 
                                    common_coord=False, 
                                    compute=True
                                    )
    bones_tracking = compute_body_segments(bp_tracking, bsegments)

    raise NotImplementedError("Find a way to save this to file")