from collections import namedtuple

from vgatPAG.database.db_tables import *

from Analysis.utils import (
            get_mouse_session_data,
            get_shelter_threat_trips,
            get_session_stimuli_frames,
            pxperframe_to_cmpersec,
            compute_stationary_vs_locomoting)
from Analysis.colors import *

px_to_cm = 13.9

# Get all mice
mice = Mouse.fetch("mouse")

# Get all sessions
sessions = {m:(Session & f"mouse='{m}'").fetch("sess_name") for m in mice}

mouse_sessions = []
for mouse in mice:
    for sess in sessions[mouse]:
        mouse_sessions.append((mouse,
                    sess, f'{mouse}-{sess}'))

# Get the recordings for each session
recordings = {m:{s:(Recording & f"sess_name='{s}'" & f"mouse='{m}'").fetch(as_dict=True) for s in sessions[m]} for m in mice}
recording_names = {m:{s:list((Recording & f"sess_name='{s}'" & f"mouse='{m}'").fetch('rec_name')) for s in sessions[m]} for m in mice}

# Get fps for each session
sessions_fps = {m:{s:Recording().get_recording_fps(mouse=m, 
                    sess_name=s, rec_name=recording_names[m][s][0]) 
                    for s in sessions[m]} 
                    for m in mice}


def print_recordings_tree():
    for _mouse in mice:
        print(f'\n{_mouse}:')
        for _session in sessions[_mouse]:
            print(f"      |---{_session}")

            recs = Recording().get_sessions_recordings(_mouse, _session)
            print("      |       |--", end='')
            print(*recs, sep=" || ")     


# Get stimuli
stimstuple = namedtuple('stimuli', 'vis aud all')
stimuli = {}
clean_stimuli = {}
for mouse in mice:
    for sess in sessions[mouse]:
        vis, aud = get_session_stimuli_frames(mouse, sess, clean=False)
        stimuli[f'{mouse}-{sess}'] = stimstuple(vis, aud, vis+aud)

        vis, aud = get_session_stimuli_frames(mouse, sess, clean=True)
        clean_stimuli[f'{mouse}-{sess}'] = stimstuple(vis, aud, vis+aud)
