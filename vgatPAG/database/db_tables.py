import datajoint as dj
import numpy as np
from pathlib import Path
from rich.prompt import IntPrompt
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
from vgatPAG.database.dj_config import start_connection, dbname, manual_insert_skip_duplicate
from pyinspect import install_traceback
install_traceback(keep_frames=1, relevant_only=True)

from fcutils.file_io.io import open_hdf
from fcutils.maths.utils import derivative, rolling_mean
from fcutils.video.utils import get_cap_from_file, get_video_params

schema = start_connection()


fld = Path('D:\\Dropbox (UCL)\\Project_vgatPAG\\analysis\\doric\\VGAT_summary\\temp_tagged-mp4')


@schema
class Mouse(dj.Manual):
    definition = """
        mouse: varchar(64)
    """

    def populate(self):
        mice_subs = [f for f in fld.glob('*') if f.is_dir()]
        mice = [f.name for f in mice_subs]
        for mouse in mice:
            manual_insert_skip_duplicate(self, {'mouse':mouse})

@schema
class Experiment(dj.Imported):
    definition = """
        # One recording on one day
        -> Mouse
        date: varchar(32)
        rec: int
        name: varchar(128)
        ---
        hdf_path: varchar(256)
        video_path: varchar(256)
        video_fps: int
        is_ca_rec: longblob
        ca_rec_starts: longblob
        ca_rec_ends: longblob
    """

    def _make_tuples(self, key):
        sub =  [f for f in fld.glob('*') if f.is_dir() and key['mouse'] in str(f)][0]

        hdfs = sorted([f for f in sub.glob('*.hdf5') if 'Fiji-tag' in f.name])
        exp_names = [f.name.split('_Fiji')[0] for f in hdfs]

        exp_data = {}
        for exp in exp_names:
            h = [h for h in hdfs if exp in str(h)]
            v = [f for f in sub.glob('*.mp4') if exp in str(f)]

            if len(h)!=1 or len(v)!=1:
                continue
                # raise ValueError(f'h:{h}\nv:{v}')

            exp_data[exp] = dict(hdf=h[0], video=v[0])

        for name, data in exp_data.items():
            if '_t' in name:
                splitter = '_t'
            else:
                splitter = '_v'

            _, _, _, fps, _ = get_video_params(get_cap_from_file(str(data['video'])))

            try:
                f, keys, subkeys, allkeys = open_hdf(str(data['hdf']))
            except Exception as e:
                print(f'Failed to open AI file: {data["hdf"].name}:\n{e}')
                return

            roi = [k for k in keys if 'Fiji_ROI' in k][0]
            sig = f[roi][()]
            is_rec = np.zeros_like(sig)
            is_rec[sig>0] = 1

            rec_starts = np.where(derivative(is_rec) > 0)[0]
            rec_ends = np.where(derivative(is_rec) < 0)[0]

            ekey = key.copy()
            ekey['date'] = name.split('_')[0]
            ekey['rec'] = int(name.split(splitter)[1][0])
            ekey['name'] = name
            ekey['hdf_path'] = str(data['hdf'])
            ekey['video_path'] = str(data['video'])
            ekey['video_fps'] = fps
            ekey['is_ca_rec'] = is_rec
            ekey['ca_rec_starts'] = rec_starts
            ekey['ca_rec_ends'] = rec_ends
            manual_insert_skip_duplicate(self, ekey)

@schema
class CaFPS(dj.Imported):
    definition = """
        -> Experiment
        --- 
        ca_fps: int

    """

    def _make_tuples(self, key):
        key['ca_fps'] = IntPrompt.ask(f'\nWhats the framerate for: {key["mouse"]} {key["date"]}?')
        manual_insert_skip_duplicate(self, key)
            

@schema
class Roi(dj.Imported):
    definition="""
        -> Experiment
        id: varchar(32)
        ---
        raw: longblob
        filtered: longblob
        dff: longblob
        dff_percentile: int
    """

    dff_percentile = 30

    @staticmethod
    def percfilt(sig, window=10, percentile=10):
        return np.array(pd.Series(sig).rolling(window=window, min_periods=1).apply(lambda x: np.percentile(x, percentile), raw=False))

    @staticmethod
    def lowpass(sig, cutoff, sampling_freq, ord=2):
        w = cutoff / (sampling_freq / 2) # Normalize the frequency
        b, a = signal.butter(ord, w, 'lowpass')
        return signal.filtfilt(b, a, sig)

    @staticmethod
    def chunk_wise(sig, chunk_starts, chunk_ends, func, *args, shift=0, **kwargs):
        out = np.zeros_like(sig)
        for start, end in zip(chunk_starts, chunk_ends):
            if shift>start:
                raise ValueError('Shift cant be bigger than statt')
            out[start:end] = func(sig[start-shift:end], *args, **kwargs)[shift:]
        return out

    def _make_tuples(self, key):
        recdata = (Experiment & key).fetch('hdf_path', 'is_ca_rec', 'ca_rec_starts', 'ca_rec_ends', 'video_fps')
        starts, ends = recdata[2][0], recdata[3][0]
        fps = recdata[4][0]

        f, keys, subkeys, allkeys = open_hdf(str(recdata[0][0]))

        for roi in [k for k in keys if 'Fiji_ROI' in k]:
            sig = f[roi][()]
            # Remove fast noise
            denoised = self.chunk_wise(sig, starts, ends, self.lowpass, 1, fps)

            # Remove slow fluctuations
            slow = self.chunk_wise(denoised, starts, ends, self.percfilt, window=10 * fps, percentile=10)

            noslow = sig-slow

            # DFF

            # Store


            f, ax = plt.subplots()
            ax.plot(sig, label='sig')
            # ax.plot(denoised, label='denoised')
            # ax.plot(slow, label='slow oscillations')
            ax.plot(noslow, label='slow removed and denoised')

            ax.legend()
            ax.set(xlim=[0, 10000])

            plt.show()
