import numpy as np
from pathlib import Path
from rich.progress import track
from vgatPAG.database.db_tables import Roi, Sessions


main_fld = Path('D:\\Dropbox (UCL)\\Project_vgatPAG\\analysis\\doric\\VGAT_summary\\temp_tagged-mp4')

sessions = (Sessions * Sessions.Tracking).fetch(as_dict=True)

for sess in track(sessions):
    mouse, date = sess['mouse'], sess['date']

    fld = main_fld / mouse / date
    fld.mkdir(exist_ok=True)

    np.save(fld/'xpos.npy', sess['x'])
    np.save(fld/'ypos.npy', sess['y'])
    np.save(fld/'speed.npy', sess['s'])

    # Get ROIs
    rois = (Roi & f'mouse="{mouse}"' & f'date="{date}"').fetch(as_dict=True)

    for roi in rois:
        np.save(fld / f'{roi["id"]}_dff.npy', roi['dff'])
        np.save(fld / f'{roi["id"]}_raw.npy', roi['raw'])
        np.save(fld / f'{roi["id"]}_slow_dff.npy', roi['slow_dff'])
        np.save(fld / f'{roi["id"]}_zscore.npy', roi['zscore'])