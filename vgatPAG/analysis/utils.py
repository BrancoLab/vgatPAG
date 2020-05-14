import numpy as np

from vgatPAG.database.db_tables import *


def get_mouse_session_data(mouse, session, sessions):
    if session not in sessions[mouse]: raise ValueError('Invalide session/mouse combo')

    # Get individual recording's within the session, and the calcium data for each
    recs = Recording().get_sessions_recordings(mouse, session)
    roi_ids, roi_sigs, nrois = Roi().get_sessions_rois(mouse, session)
    roi_ids = list(roi_ids.values())[0]

    # print(f'\nNumber of rois per recording: | should be the same for each recording\n  {nrois}')

    # Concatenate calcium signals
    _signals = [[] for i in np.arange(nrois[recs[0]])]
    is_rec = []

    for rec in recs:
        is_rec.append((TiffTimes & f"rec_name='{rec}'").fetch1("is_ca_recording"))

        sigs = roi_sigs[rec]
        for r, roi in enumerate(sigs):
            _signals[r].append(roi)

    is_rec = np.hstack(is_rec)
    signals = [np.hstack(s).T for s in _signals]
    _nrois = len(signals)

    # Get tracking data 
    body_tracking, ang_vel, speed, shelter_distance = [], [], [], []
    for rec in recs:
        bt, av, s, sd = Trackings().get_recording_tracking_clean(mouse_name=mouse, sess_name=session, rec_name=rec)
        body_tracking.append(bt)
        ang_vel.append(av)
        speed.append(s)
        shelter_distance.append(sd)

    body_tracking = np.hstack(body_tracking)
    ang_vel = np.hstack(ang_vel).T
    speed = np.hstack(speed).T
    shelter_distance = np.hstack(shelter_distance).T

    tracking = pd.DataFrame(dict(x = body_tracking[0, :], y=body_tracking[1, :], s=body_tracking[1, :])).interpolate()

    # Make sure stuff has right length
    clean_signal = []
    for sig in signals:
        if len(sig) < len(tracking):
            s = np.zeros((len(tracking)))
            s[:len(sig)] = sig
            clean_signal.append(s)
        elif len(sig) == len(tracking):
            clean_signal.append(sig)
        else:
            raise ValueError
    
    if len(is_rec) < len(tracking):
        a = np.zeros((len(tracking)))
        a[:len(is_rec)] = is_rec
        is_rec = a

    return tracking, ang_vel, speed, shelter_distance, clean_signal, _nrois, is_rec