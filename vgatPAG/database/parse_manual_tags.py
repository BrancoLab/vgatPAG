import sys
sys.path.append('./')

from fcutils.file_io.io import open_hdf, load_yaml
import pandas as pd
from vgatPAG.database.db_tables import Sessions
from pathlib import Path
from fcutils.video.utils import trim_clip
import numpy as np

def parse_manual_tags_file():
    tags_names = ['VideoTag_A', 'VideoTag_B', 'VideoTag_C', 'VideoTag_D',
                    'VideoTag_E', 'VideoTag_F', 'VideoTag_G', 'VideoTag_H', 'VideoTag_L']

    # filepath = '/Users/federicoclaudi/Dropbox (UCL - SWC)/Project_vgatPAG/analysis/doric/VGAT_summary/VGAT_summary_tagData.hdf5'
    filepath = 'D:\\Dropbox (UCL - SWC)\\Project_vgatPAG\\analysis\\doric\\VGAT_summary\\VGAT_summary_tagData.hdf5'

    f, keys, subkeys, all_keys = open_hdf(filepath)
    events_types = subkeys['all']

    tags = dict(
        mouse = [],
        sess_name = [],
        rec_number = [],
        event_type = [],
        tag_type = [],
        frame = [],
        session_frame = [],
        session_stim_frame = [],
        stim_frame=[],
    )

    def add_entry_tags(sess, events, event_type, mouse, sess_name, rec_number, rec_idx):        
        # Get total number of frames
        n_frames = sess.frames_shift


        for event in list(events): # loop over frame_xxx video
            if 'TagData' not in list(events[event]):
                continue

            for tn in tags_names: # loop over each manual tag
                if tn not in list(events[event]['TagData']['TagTimes_absolute']): continue
                for time in list(events[event]['TagData']['TagTimes_absolute'][tn]): # loop over the time stamps for the tag
                    # The time is in ms from 20s before the start of the stim
                    # turn it to number of frames from start
                    stimframe = int(event.split('_')[-1])
                    framen = int(((time / 1000)-20) * 30) + stimframe

                    # CHECK IF WE should excude this
                    if 'B' in tn:
                        ignore_trials = load_yaml('vgatPAG\database\manual_tags_exclude.yml')
                        name = f'{mouse}_{sess_name}_t{rec_number}_{framen}'
                        if name in ignore_trials:
                            print('excluding trials from ', name)
                            continue

                    # Get the frame number from the start of the session
                    if mouse == 'BF164p1' and sess_name == '19JUN18':
                        rec_number = 1
                        rec_idx = 1

                    if rec_idx == 0:
                        framen_sess = framen
                        session_stim_frame = stimframe
                    else:
                        framen_sess = framen + n_frames[rec_idx-1]
                        session_stim_frame = stimframe + n_frames[rec_idx-1]

                    if framen_sess > np.sum(n_frames):
                        R = (Recording & f"sess_name='{sess_name}'" & f"mouse='{mouse}'").fetch("rec_name")
                        raise ValueError('Something went wrong when computing session frame number.\n'+
                                        f'{R}')

                    tags['mouse'].append(mouse)
                    tags['sess_name'].append(sess_name)
                    tags['rec_number'].append(rec_number)
                    tags['event_type'].append(event_type)
                    tags['tag_type'].append(tn)
                    tags['frame'].append(framen)
                    tags['session_frame'].append(framen_sess)
                    tags['stim_frame'].append(stimframe)
                    tags['session_stim_frame'].append(session_stim_frame)


    # loop over each session
    for i, sess in pd.DataFrame(Sessions().fetch()).iterrows():
        mouse = sess.mouse
        date = sess.date
        name = mouse+'_'+date

        # loop over each event type (e.g. UltrasoundLoom)
        for ev in events_types:
            # Get entries for the session (1 per recording)
            entries = [k for k in list(f['all'][ev]) if name in k]

            # Extract tags
            if len(entries) == 1:
                entry = entries[0]
                if '_t' in entry:
                    idx = entry.index('_t')
                    tnum = int(entry[idx+2:idx+3])
                else:
                    tnum = 0
                add_entry_tags(sess, f['all'][ev][entries[0]],
                                ev, mouse, date, tnum, 0)
            elif entries:
                trials = []
                for entry in entries:
                    idx = entry.index('_t')
                    trials.append(int(entry[idx+2:idx+3]))
                trials = list(set(trials))
                
                if len(set(trials)) == 1:
                    if mouse != 'BF164p1':
                        raise NotImplementedError('This has only been tested for one mouse')
                    t = entry[idx:idx+3]
                    recnames = (Recording & f"sess_name='{date}'" & f"mouse='{mouse}'").fetch('rec_name')
                    tridx = [j for j,name in enumerate(recnames) if t in name]
                    if len(tridx) != 1:
                        raise ValueError

                    for entry in entries:
                        add_entry_tags(sess, f['all'][ev][entry],
                                ev, mouse, date, trials[0], tridx[0]+1)
                else:
                    for nt, (entry, tnum) in enumerate(zip(entries, trials)):
                        add_entry_tags(sess, f['all'][ev][entry],
                                ev, mouse, date, tnum, nt)


    return pd.DataFrame(tags)

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
