import sys
sys.path.append('./')

from fcutils.file_io.io import open_hdf, load_yaml
import pandas as pd
from vgatPAG.database.db_tables import Sessions
from pathlib import Path
from fcutils.video.utils import trim_clip
import numpy as np

def make_videos():
    # savefld = Path('/Users/federicoclaudi/Dropbox (UCL - SWC)/Project_vgatPAG/analysis/doric/Fede/manual_tag_videos')
    savefld = Path(r'D:\Dropbox (UCL - SWC)\Project_vgatPAG\analysis\doric\Fede\manual_tag_videos')

    tag_type = 'VideoTag_B'
    loom_escapes = tags.loc[(tags.event_type == 'Loom_Escape')&(tags.tag_type == tag_type)]
    # loop over each mouse
    for mouse in loom_escapes.mouse.unique():
        # loop over each session
        for sess in loom_escapes.loc[loom_escapes.mouse==mouse].sess_name.unique():
            

            sess_data = loom_escapes.loc[(loom_escapes.mouse==mouse)&(loom_escapes.sess_name==sess)]

            tracking = pd.DataFrame((Trackings * Trackings.BodyPartTracking & 'bp="body"' 
                                & f'mouse="{mouse}"' & f"sess_name='{sess}'"))
            
            # look over each recording
            for n, recn in enumerate(sess_data.rec_number.unique()):
                events = sess_data.loc[sess_data.rec_number == recn]

                # Get recording video:
                if mouse == 'BF164p1' and sess == '19JUN18':
                    idx = recn
                else:
                    idx = n
                recv = (Recording & f'mouse="{mouse}"' & f"sess_name='{sess}'" & f"rec_n={idx}").fetch1('videofile')
                recv = Path(get_session_folder(mouse, sess)) / recv
                
                # loop over each event
                for frame in events.frame.values:
                    trim_clip(str(recv), 
                                str(savefld / f'{mouse}_{sess}_t{recn}_{frame}_{tag_type}.mp4'),
                                frame_mode=True,
                                start_frame = frame - 30 * 2,
                                stop_frame = frame + 30*9,
                            )



if __name__ == '__main__':
    tags = parse_manual_tags_file()
    tags.to_hdf('vgatPAG/database/manual_tags.h5', key='h5')

    # make_videos()
