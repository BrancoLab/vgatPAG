import numpy as np
from fcutils.maths.filtering import smooth_hanning

from vgatPAG.database.db_tables import *
from scipy.ndimage.filters import gaussian_filter1d
from behaviour.utilities.signals import get_times_signal_high_and_low

# ------------------------------ Computing stuff ----------------------------- #
def compute_stationary_vs_locomoting(speed, **kwargs):
    """
        Given a 1d numpy array with speed data, it returns a 1d numpy array which
        is 1 when the mouse is locomoting and 0 otherwise. 
        This is done by:
            - convolving the speed trace with a gaussian to smooth
            - Getting times of high vs low speed 
            - ignoring brief intervals of locomotion/stationary
    """

    if len(speed.shape) > 1: raise ValueError('Invalid input')

    guassian_sigma = kwargs.pop('gaussian_sigma', 6) # for smoothing
    speed_threshold = kwargs.pop('speed_threshold', 1.5) # speed > th is locomotion
    min_event_duration = kwargs.pop('min_event_duration', 10) #  events shorter than this are ignored

    # Smooth, threshold and get events
    smoothed = gaussian_filter1d(speed, guassian_sigma, mode='constant', cval=0)

    is_moving = np.ones_like(smoothed)
    is_moving[smoothed < speed_threshold] = 0

    starts, ends = get_times_signal_high_and_low(is_moving, th=.5)
    ends = [e for e in ends if e > starts[0]]

    # Ignore fast locomotion events
    for start, end in zip(starts, ends):
        if end - start < min_event_duration:
            is_moving[start:end+1] = 0

    # Ignore fast stationary events
    for start, end in zip(starts[1:], ends):
        if start-end < min_event_duration:
            is_moving[end:start+1] = 1

    return is_moving


# ----------------------------------- Misc ----------------------------------- #

def pxperframe_to_cmpersec(pxperframe, fps=30, pxpercm=13.9):
    return (pxperframe/pxpercm) * fps


# ------------------------------- Data getters ------------------------------- #

def get_session_stimuli_frames(mouse, sess, clean=True):
    """
        Returns the frame number of each stimulus within one experiment
        When >1 recordings where done within the same experiment
        it returns the comulative frame number (i.e. the frame
        number from the start of the first recording not from the
        the strt of the recording the stim happend in).
        So it works well with get_mouse_session_data below.
    """
    recs = Recording().get_sessions_recordings(mouse, sess)

    cum_nframes = 0
    astims, vstims = [], []
    for rec in recs:
        kwargs = dict(sess_name=sess, mouse=mouse, rec_name = rec)
        
        vis, aud = Recording().get_recording_stimuli(clean=clean, **kwargs)

        astims.extend([s+cum_nframes for s in aud])
        vstims.extend([s+cum_nframes for s in vis])

        nframes = (Recording & kwargs).fetch1('n_frames')
        cum_nframes += nframes

    return vstims, astims


def get_mouse_session_data(mouse, session, sessions, hanning_window=6):
    """
        Fetches tracking and calcium data for one mouse in one experiment
        concatenating across all recordings within that session.
    """
    if session not in sessions[mouse]: raise ValueError('Invalide session/mouse combo')

    # Get individual recording's within the session, and the calcium data for each
    recs = Recording().get_sessions_recordings(mouse, session)
    roi_ids, roi_sigs, nrois = Roi().get_sessions_rois(mouse, session)
    roi_ids = list(roi_ids.values())[0]

    # Concatenate calcium signals from each recording within the experiment
    _signals = [[] for i in np.arange(nrois[recs[0]])]
    is_rec = []

    for rec in recs:
        is_rec.append((TiffTimes & f"rec_name='{rec}'").fetch1("is_ca_recording"))

        sigs = roi_sigs[rec]
        for r, (roi, rid) in enumerate(zip(sigs, roi_ids)):
            _signals[r].append(roi)

    is_rec = np.hstack(is_rec)
    signals = [np.hstack(s).T for s in _signals]
    _nrois = len(signals)

    # Get tracking data from all recordings and concatenate
    body_tracking, ang_vel, speed, shelter_distance = [], [], [], []
    for rec in recs:
        bt, av, s, sd = Trackings().get_recording_tracking_clean(mouse_name=mouse, sess_name=session, rec_name=rec)
        body_tracking.append(bt)
        ang_vel.append(av)
        speed.append(s)
        shelter_distance.append(sd)

    body_tracking = np.hstack(body_tracking) # ! concatenating
    ang_vel = np.hstack(ang_vel).T
    speed = np.hstack(speed).T
    shelter_distance = np.hstack(shelter_distance).T

    tracking = pd.DataFrame(dict(x = body_tracking[0, :], y=body_tracking[1, :], s=body_tracking[1, :])).interpolate()

    # Make sure stuff has right length
    tot_frames = np.sum([
            (Recording & f'rec_name="{r}"').fetch1('n_frames') for r in recs
    ])
    if len(tracking) != tot_frames:
        raise ValueError(f'Incorrect # frames for tracking: {len(tracking)} instead of {tot_frames}')


    clean_signal = []
    for sig in signals:
        if hanning_window:
            sig = smooth_hanning(sig, window_len=hanning_window)[:-hanning_window-1]


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

    # Compute dffs
   
    dffs = []
    for rid, sig in zip(roi_ids, clean_signal):

        try:
            dffth = (RoiDFF & f"roi_id='{rid}'" & f"sess_name='{session}'" & f"mouse='{mouse}'").fetch1('dff_th')
        except Exception as e:
            print(f'Failed to get mouse session data: {e}')
            return None,  None,  None,  None,  None,  None,  None

        if hanning_window:
            dffs.append(smooth_hanning((sig-dffth)/dffth, window_len=hanning_window)[:-hanning_window-1])
        else:
            dffs.append((sig-dffth)/dffth,)

    dffs = [df * is_rec[:len(df)] for df in dffs]
    clean_signal = [df * is_rec[:len(df)] for df in clean_signal]

    return tracking, ang_vel, speed, shelter_distance, dffs, clean_signal, _nrois, is_rec, roi_ids


def get_shelter_threat_trips(data, shelter_x=400, threat_x=800, only_recording_on=True, 
                stimuli=None, min_frames_after_stim=10*30, 
                mean_speed_th = 2):
    """
        Returns the timepoints when the mouse moves between the shelter and the threat

        :param data, pandas.DataFrane with ['x'] location as one of the columns
        :param shelter_x: int, when x < this mouse is considered in the shelter
        :param threat_x: int, when x > this mouse is considered in the threat area
        :param only_recording_on: bool. If true only trips when the recording was ON
                throughout are considered. If using this the data DF needs to have a 
                ['isrec'] column
        :param stimuli: optional, a list with the frame numner of each stim in the session
        :param min_frames_after_stim: int if stimuli are passed, when getting threat->shelter
                trips, onlyt those trips where at least min_frames_after_stim frames
                have passed between the last stimulus and the threat exit . 
        :param mean_speed_th: float. if not None, only trips with average speed > mean_speed_th 
                are kept

    """
    # Get times when mouse enters/exits shelter and threat
    in_shelter = np.zeros_like(data.x)
    in_shelter[data.x < shelter_x] = 1


    in_threat = np.zeros_like(data.x)
    in_threat[data.x > threat_x] = 1


    shelter_enter, shelter_exit = get_times_signal_high_and_low(in_shelter, th=.5)
    threat_enter, threat_exit = get_times_signal_high_and_low(in_threat, th=.5)

    # Get s -> t events
    outtrips = []
    all_events = sorted(list(shelter_exit) + list(threat_enter))
    
    # ------------------------ Get shelter to threat trips ----------------------- #    
    # For everytime the mouse leaves the shelter
    for ext in shelter_exit:
        # Check if the next event is the mouse entering the threa area
        nxt = [ev for ev in all_events if ev > ext]
        if nxt:
                if nxt[0] in threat_enter:
                    # Make sure that recording was on throughout the trip
                    if only_recording_on:
                        not_recording = [0 for i in data.isrec[ext:nxt[0]].values if not i]
                        if not_recording:
                            continue

                    # Check if mean speed > threshold
                    if mean_speed_th is not None:
                        if data[ext:nxt[0]].s.mean() < mean_speed_th:
                            continue

                    # Add to records
                    outtrips.append((ext, nxt[0]))

    # ------------------------ Get threat to shelter trips ----------------------- #

    intrips = []
    all_events = sorted(list(threat_exit) + list(shelter_enter))
    
    # For everytime the mouse leaves the threat area
    for ext in threat_exit:
        # If the next event is the moues entering the shelter area
        nxt = [ev for ev in all_events if ev > ext]
        if nxt:
                if nxt[0] in shelter_enter:

                    # Make sure that recirding was on throughout the trip
                    if only_recording_on:
                        not_recording = [0 for i in data.isrec[ext:nxt[0]].values if not i]
                        if not_recording:
                            continue
                    
                    # Exclude trips that happened right after a stimulus
                    if stimuli is not None:
                        last_stim = [s for s in stimuli if s < ext]
                        if last_stim:
                            if ext - last_stim[-1] < min_frames_after_stim:
                                continue

                    # Check if mean speed > threshold
                    if mean_speed_th is not None:
                        if data[ext:nxt[0]].s.mean() < mean_speed_th:
                            continue

                    intrips.append((ext, nxt[0]))


    return outtrips, intrips

