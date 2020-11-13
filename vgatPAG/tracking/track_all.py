"""
    Uses deeplabcut to track all the videos in dataset
"""

import sys
sys.path.append("./")
import os 
from pathlib import Path

import deeplabcut as dlc

from fcutils.file_io.utils import get_subdirs, listdir, get_file_name

from behaviour.tracking.tracking import prepare_tracking_data, compute_body_segments

from vgatPAG.paths import dlc_config_file


bsegments = [ # Body segments used to get stuff like angular velocity, orientation etc
    ['snout', 'neck'],
    ['neck', 'body'],
    ['body', 'tail_base']
]


fld = Path('D:\\Dropbox (UCL)\\Project_vgatPAG\\analysis\\doric\\VGAT_summary\\temp_tagged-mp4')

mice_subs = [f for f in fld.glob('*') if f.is_dir()]


videos_to_process = []
for sub in mice_subs:
    vids = [f for f in sub.glob('*.mp4')]
    h5s = [f.name.split('DLC')[0] for f in sub.glob('*.h5')]
    vids = [v for v in vids if v.name.split('.mp4')[0] not in h5s]
    videos_to_process.extend(vids)
videos_to_process = [str(v) for v in videos_to_process]

# Start tracking
print(f"Found {len(videos_to_process)} videos not tracked:")
print("Firing DLC up...")

dlc.analyze_videos(dlc_config_file, 
                videos_to_process, 
                save_as_csv=False,
                dynamic=(True, 0.5, 100))



