import numpy as np
from fcutils.maths.utils import derivative

from vgatPAG.database.db_tables import RoiDFF

def get_tiff_starts_ends(is_rec):
    '''
        Gets the start and end time of doric chunks
        for a session
    '''
    der = derivative(is_rec)
    tiff_starts = np.where(der > 0)[0]
    tiff_ends = np.where(der < 0)[0]

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