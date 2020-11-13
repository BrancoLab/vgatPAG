import numpy as np
from fcutils.maths.utils import derivative, rolling_mean
from fcutils.maths.filtering import butter_lowpass_filter

from vgatPAG.database.db_tables import RoiDFF

def get_tiff_starts_ends(is_rec):
    '''
        Gets the start and end time of doric chunks
        for a session
    '''
    der = derivative(is_rec)
    tiff_starts = np.where(der > 0)[0] + 1
    tiff_ends = np.where(der < 0)[0] + 1

    return tiff_starts, tiff_ends

def percentile(sig, p=None):
    if p is None:
        p = RoiDFF.DFF_PERCENTILE
    return np.percentile(sig, p)

def get_doric_chunks(tiff_starts, tiff_ends, start):
    # get start and end of doric chunk around a frame
    tstart = [tf for tf in tiff_starts if tf < start][-1]
    tend = [te for te in tiff_ends if te > start][0]

    if tstart > start or tend < start:
        raise ValueError
    return tstart, tend

def get_doric_chunk_baseline(tiff_starts, tiff_ends, start, signal, p=None):
    # baseline raw for whole doric chunk given a start frame
    tf, te = get_doric_chunks(tiff_starts, tiff_ends, start)

    if te<tf: raise ValueError
    return percentile(signal[tf:te], p=p)

def get_chunked_dff(sig, tiff_starts, tiff_ends):
    dff = np.zeros_like(sig)

    for start, end in zip(tiff_starts, tiff_ends):
        th = percentile(sig[start:end])
        dff[start:end] = (sig[start:end] - th)/th
        dff[start] = dff[start+1]
        dff[end-3:end] = dff[end-4]
    return dff

def get_chunk_rolling_mean_subracted_signal(sig, tiff_starts, tiff_ends, window=100):
    """
        Takes a roi RAW signal and computes a rolling mean of each signal's chunk.
        Then it subtracts this smoothed signal from the raw to get only high
        frequency noise.
    """
    pad = 1
    smoothed = np.zeros_like(sig)

    for start, end in zip(tiff_starts, tiff_ends):
        smoothed[start:end] = rolling_mean(sig[start:end], window)
        # smoothed[start:end] = butter_lowpass_filter(sig[start:end], .1,  100)
        smoothed[start:start+5] = sig[start:start+5]
        smoothed[end-5:end] = sig[end-5:end]


    diff = sig - smoothed

    return diff, smoothed