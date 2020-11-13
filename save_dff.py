import numpy as np
from pathlib import Path
from rich.progress import track
from Analysis import (
        mice,
        sessions,
        recordings,
        mouse_sessions,
        get_mouse_session_data,
)
from Analysis.tag_aligned import (
    manual_tags,
    get_tags_by,
    get_next_tag,
    get_last_tag,
)
from Analysis.misc import get_tiff_starts_ends, get_chunked_dff
from vgatPAG.database.db_tables import Recording


main_fld = Path('D:\\Dropbox (UCL)\\Project_vgatPAG\\analysis\\doric\\VGAT_summary\\temp_tagged-mp4')

for mouse, sess, sessname in track(mouse_sessions, description='saving DFFs'):
    
    # Get data
    tracking, ang_vel, speed, shelter_distance, dffs, signals, nrois, is_rec, roi_ids = \
                        get_mouse_session_data(mouse, sess, sessions)

    # Make folder to save
    fld = main_fld / mouse
    if not fld.exists():
        raise FileExistsError(f'Cant find folder {fld}')
    sessfld = fld / f'{sess}_DFF'
    sessfld.mkdir(exist_ok=True)

    # loop over rois
    for n, (sig, rid) in enumerate(zip(signals, roi_ids)):
        # Get chunks start end times
        tiff_starts, tiff_ends = get_tiff_starts_ends(is_rec)

        # get chunked dff
        dff = get_chunked_dff(sig, tiff_starts, tiff_ends)

        np.save(sessfld/f'{sess}_{mouse}_{rid}_dff.npy', dff)

