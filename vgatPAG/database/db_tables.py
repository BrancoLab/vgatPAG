import datajoint as dj
import numpy as np
from pathlib import Path
from rich.prompt import IntPrompt
from scipy import signal
from scipy.stats import zscore
from rich.progress import track
import pandas as pd
import matplotlib.pyplot as plt
from vgatPAG.database.dj_config import start_connection, dbname, manual_insert_skip_duplicate
from pyinspect import install_traceback
install_traceback(keep_frames=2, relevant_only=True)

from fcutils.file_io.io import open_hdf
from fcutils.maths.utils import derivative, rolling_mean
from fcutils.video.utils import get_cap_from_file, get_video_params
from fcutils.plotting.utils import save_figure
from _tracking import prepare_tracking_data

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

        print(f"Processing {key} - {sub}")
        # print(f"Found subfolders {sub}")

        hdfs = sorted([f for f in sub.glob('*.hdf5') if 'Fiji-tag' in f.name and 'forFede' not in f.name])
        exp_names = [f.name.split('_Fiji')[0] for f in hdfs]
        # print(f"Found HDF files: {hdfs}")

        exp_data = {}
        for exp in exp_names:
            h = [h for h in hdfs if exp in str(h)]
            v = [f for f in sub.glob('*.mp4') if exp in str(f)]

            if len(h)!=1 or len(v)!=1:
                print(f"Mouse {key['mouse']}, exp: {exp} --- number of videos {len(v)} doesnt match number of HDFs {len(h)} | {len(hdfs)}")
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
class Trackings(dj.Imported):
    definition = """
        -> Experiment
    """
    bparts = ['snout', 'left_ear', 'right_ear', 'neck', 'body', 'tail_base']
    bsegments = {'head':('snout', 'neck'), 
                'upper_body':('neck', 'body'),
                'lower_body':('body', 'tail_base'),
                'whole_body':('snout', 'tail_base')}
    main_fld = Path('D:\\Dropbox (UCL)\\Project_vgatPAG\\analysis\\doric\\VGAT_summary\\temp_tagged-mp4')

    class BodyPartTracking(dj.Part):
        definition = """
            -> Trackings
            bp: varchar(64)
            ---
            x: longblob
            y: longblob
            speed: longblob
            dir_of_mvmt: longblob
            angular_velocity: longblob
        """

    def smooth_scale(self, data, is_speed = False, is_deg=False):
        # Smooth with rolling mean
        data = rolling_mean(data, 3)

        if not is_deg:
            if is_speed:
                # px/frame -> cm/s
                data = (data/13.9) * 30 # x / px per cm * fps
            else:
                # px -> cm
                data = data / 13.9
        else:
            # deg/frame -> deg/s
            data = data * 30
        return data

    # Populate method
    def _make_tuples(self, key):
        fld = self.main_fld/key["mouse"]
        tracking_file = [f for f in fld.glob('*.h5') if key['name'] in f.name]
        if len(tracking_file) != 1:
            raise ValueError(f'found too many or too few: {tracking_file}')

        # Insert entry into main class
        self.insert1(key)

        # Load and clean tracking data
        bp_tracking = prepare_tracking_data(str(tracking_file[0]), 
                                        median_filter=True,
                                        likelihood_th=0.9,
                                        fisheye=False, common_coord=False, compute=True)

        # Insert into the bodyparts tracking
        for bp, tracking in bp_tracking.items():
            bp_key = key.copy()
            bp_key['bp'] = bp
            
            bp_key['x'] = self.smooth_scale(tracking.x.values, is_speed=False)
            bp_key['y'] = self.smooth_scale(tracking.y.values, is_speed=False)
            bp_key['speed'] = self.smooth_scale(tracking.speed.values, is_speed=True)
            bp_key['dir_of_mvmt'] = rolling_mean(tracking.direction_of_movement.values, 3)
            bp_key['angular_velocity'] = self.smooth_scale(tracking.angular_velocity.values, is_deg=True)

            self.BodyPartTracking.insert1(bp_key)

@schema
class Sessions(dj.Computed):
    definition = """
        # It concatenates a mouse experiment's that occurred on the same day
        -> Mouse
        date: varchar(32)
        ---
        video_fps: int
        is_ca_rec: longblob
        ca_rec_starts: longblob
        ca_rec_ends: longblob
        exps_nframes: longblob
        frames_shift: longblob  # how many frames to shift stuff by
    """  

    class RoiTrace(dj.Part):
        definition = """
            # Concatenated ROI trace across experiments
            -> Sessions
            roi_name: varchar(128)
            ---
            sig: longblob
        """

    class Tracking(dj.Part):
        definition = """
            # Concatenated tracking across experiments
            -> Sessions
            ---
            x: longblob
            y: longblob
            s: longblob
            dir_of_mvmt: longblob
        """

    
    def _make_tuples(self, key):
        # Get all experiments for this mouse, by date
        exps = pd.DataFrame((Experiment & key).fetch(as_dict=True))
        if len(exps) == 0:
            return

        # Loop over dates
        for date in exps.date.unique():
            es = exps.loc[exps.date == date].sort_values('name', axis=0)

            # get stuff
            n_frames = [len(r.is_ca_rec) for i,r in es.iterrows()]
            frames_shift = [0] + list(np.cumsum(n_frames[:-1]))
            if len(es.video_fps.unique()) >1:
                raise ValueError('More than one video fps')

            is_ca_rec = np.concatenate(es.is_ca_rec.values)
            ca_rec_starts = np.concatenate([np.array(ca)+frames_shift[n] for n,ca in enumerate(es.ca_rec_starts.values)])
            ca_rec_ends = np.concatenate([np.array(ca)+frames_shift[n] for n,ca in enumerate(es.ca_rec_ends.values)])

            # f, ax = plt.subplots()
            # ax.plot(is_ca_rec)
            # for s,e in zip(ca_rec_starts, ca_rec_ends):
            #     ax.axvline(s, color='g')
            #     ax.axvline(e, color='m')
            # plt.show()

            # Check that ROIs match
            roi_data = [open_hdf(f) for f in es.hdf_path]
            rois = sorted([k for k in roi_data[0][1] if 'Fiji_ROI' in k ])
            if len(roi_data) > 1:
                for (_, keys, _, _) in roi_data[1:]:
                    for k in keys:
                        if 'Fiji_ROI' in k and k not in rois:
                            raise ValueError('Unrecognized ROI')

            # Stack ROIs
            rois_stacks = {}
            for roi in rois:
                rois_stacks[roi] = np.concatenate([f[roi][()] for f, _, _, _ in roi_data])

            # Stack tracking
            trackings = []
            for i,s in es.iterrows():
                name = s['name']
                trackings.append((Trackings * Trackings.BodyPartTracking & key 
                                        & 'bp="body"' & f'date="{s.date}"' & f'name="{name}"').fetch('x', 'y', 'speed', 'dir_of_mvmt', as_dict=True)[0])

            keys = list(trackings[0].keys())
            tracking = {k: np.concatenate([t[k] for t in trackings]) for k in keys}
            

            # time to insert in the table
            mkey = key.copy()
            mkey['date'] = date
            mkey['video_fps'] = es.video_fps.values[0]
            mkey['is_ca_rec'] = is_ca_rec
            mkey['ca_rec_starts'] = ca_rec_starts
            mkey['ca_rec_ends'] = ca_rec_ends
            mkey['exps_nframes'] = n_frames
            mkey['frames_shift'] = frames_shift
            manual_insert_skip_duplicate(self, mkey)

            tkey = key.copy()
            tkey['date'] = date
            tkey['x'] = tracking['x']
            tkey['y'] = tracking['y']
            tkey['s'] = tracking['speed']
            tkey['dir_of_mvmt'] = tracking['dir_of_mvmt']
            manual_insert_skip_duplicate(self.Tracking, tkey)

            for roi, data in rois_stacks.items():
                rkey = key.copy()
                rkey['date'] = date
                rkey['roi_name'] = roi
                rkey['sig'] = data
                manual_insert_skip_duplicate(self.RoiTrace, rkey)

@schema
class Roi(dj.Imported): 
    definition="""
        -> Sessions
        id: varchar(32)
        ---
        raw: longblob
        dff: longblob
        slow_dff: longblob
        zscore: longblob
        dff_percentile: int
        slow_filter_window: int  # width in seconds of filter to remove slow effects
    """

    dff_percentile = 30
    slow_filter_window = 10  # width in seconds of the window used to filter slow transients

    @staticmethod
    def percfilt(sig, window=10, percentile=10):
        return np.array(pd.Series(sig).rolling(window=window, min_periods=1).apply(lambda x: np.percentile(x, percentile), raw=False))

    @staticmethod
    def lowpass(sig, cutoff, sampling_freq, ord=2):
        w = cutoff / (sampling_freq / 2) # Normalize the frequency
        b, a = signal.butter(ord, w, 'lowpass')
        return signal.filtfilt(b, a, sig)

    @staticmethod
    def dff(sig, perc):
        th = np.percentile(sig, perc)
        return (sig-th)/th, th

    @staticmethod
    def chunk_wise(sig, chunk_starts, chunk_ends, func, *args, shift=0, **kwargs):
        out = np.zeros_like(sig)
        for start, end in zip(chunk_starts, chunk_ends):
            if shift>start:
                raise ValueError('Shift cant be bigger than statt')
            out[start:end] = func(sig[start-shift:end], *args, **kwargs)[shift:]
        return out

    @staticmethod
    def merge_apply_split(sig, is_rec, chunk_starts, chunk_ends, func, *args, **kwargs):
        # merge signal
        out = np.zeros_like(sig)
        merged = sig[is_rec==1]

        # apply
        try:
            applied, other = func(merged, *args, **kwargs)
        except  ValueError:
            applied = func(merged, *args, **kwargs)
            other = None

        # Split again
        n = 0
        for start, end in zip(chunk_starts, chunk_ends):
            dur = end - start
            out[start:end] = applied[n:n+dur]
            n += dur
        
        return out, other


    def _make_tuples(self, key):
        rois = pd.DataFrame((Sessions * Sessions.RoiTrace & key).fetch(as_dict=True))

        for i, roi in rois.iterrows():
            sig = roi.sig
            is_rec = roi.is_ca_rec
            starts = roi.ca_rec_starts
            ends = roi.ca_rec_ends
            fps = roi.video_fps

            # remove noise
            sig = self.chunk_wise(sig, starts, ends, rolling_mean, 3)

            # whole session DFF
            dff, th = self.merge_apply_split(sig, is_rec, starts, ends, self.dff, self.dff_percentile)

            # zscore
            zscored, _ = self.merge_apply_split(dff, is_rec, starts, ends, zscore)

            # Remove slow fluctuations
            slow = self.chunk_wise(dff, starts, ends, rolling_mean, self.slow_filter_window * fps)

            # # save figure with traces
            # fig, axarr = plt.subplots(ncols=2, sharex=True, figsize=(32, 9))
            # axarr[0].plot(sig, color='k', lw=2, label='raw')
            # axarr[0].plot(is_rec, label='rec on')
            # axarr[0].axhline(th, lw=2, color='seagreen', label='F')
            # axarr[0].legend()

            # axarr[1].plot(dff, color='k', lw=3, label='dff')
            # axarr[1].plot(slow, color='salmon', lw=1, label='low pass')
            # axarr[1].legend()
            # x = np.arange(0, len(sig), 300*fps)
            # axarr[1].set(title=roi.roi_name, xticks=x, xticklabels=(x/fps).astype(np.int64), xlabel='seconds')
            # save_figure(fig, f'D:\\Dropbox (UCL)\\Project_vgatPAG\\analysis\\doric\\Fede\\dff_filtering\\{key["mouse"]}_{key["date"]}_{roi.roi_name}', verbose=False)
            # plt.close()

            # Store
            rkey = key.copy()
            rkey['id'] = roi.roi_name
            rkey['raw'] = sig
            rkey['dff'] = dff
            rkey['slow_dff'] = slow
            rkey['zscore'] = zscored
            rkey['dff_percentile'] = self.dff_percentile
            rkey['slow_filter_window'] = self.slow_filter_window
            manual_insert_skip_duplicate(self, rkey)

@schema
class ManualBehaviourTags(dj.Imported):
    definition = """
        # manual tagging of behaviour from videos by Vanessa
        -> Sessions
    """
    filepath = 'D:\\Dropbox (UCL)\\Project_vgatPAG\\analysis\\doric\\VGAT_summary\\VGAT_summary_tagData.hdf5'
    tags_names = ['VideoTag_A', 'VideoTag_B', 'VideoTag_C', 'VideoTag_D',
                        'VideoTag_E', 'VideoTag_F', 'VideoTag_G', 'VideoTag_H', 'VideoTag_L']

    class Tags(dj.Part):
        definition = """
            -> ManualBehaviourTags
            event_type: varchar(64)
            tag_type: varchar(64)
            frame: int
            session_frame: int
            ---
            stim_frame: int
            session_stim_frame: int
        """
    def _append(self, key, event_type, tag, onset_frame, stim, stim_session):
            tkey = key.copy()
            tkey['event_type'] = event_type
            tkey['tag_type'] = tag
            tkey['frame'] = onset_frame +  stim
            tkey['session_frame'] = onset_frame + stim_session
            tkey['stim_frame'] = stim
            tkey['session_stim_frame'] = stim_session

            # skip trials where cable got caught in arena
            if key['mouse'] == 'BF164p1':
                if key['date'] == '19JUN03' and stim == 19415:
                    return
            elif key['mouse'] == 'BF164p1':
                if key['date'] == '19JUN05' and stim == 95806:
                    return
                if key['date'] == '19JUN018' and stim == 32290:
                    return

            manual_insert_skip_duplicate(self.Tags, tkey)


    def process_stim_evoked(self, event_type, f, frames_shift, key, rec_n):
        for stim, entries in dict(f[event_type]).items():
            # get stim frame and relative to sess start
            stim = int(stim.split('_')[-1])
            stim_session = stim + frames_shift[rec_n]

            # get individual tags
            tags = [k for k in dict(entries).keys() if 'VideoTag_' in k]
            for tag in tags:
                # tag time in ms relative to stim onset
                onset_ms = dict(entries)[tag][()][0] - 20000

                # tag time in frame relative to stim onset
                onset_frame = int(onset_ms * 30 / 1000)

                # We got everything, add to table
                self._append(key, event_type, tag, onset_frame, stim, stim_session)


    def process_other(self, event_type, f, frames_shift, key, rec_n):
        for stim, entries in dict(f[event_type]).items():
            # get stim frame and relative to sess start
            stim = int(stim.split('_')[-1])
            stim_session = stim + frames_shift[rec_n]

            tag = 'VideoTag_B'
            # the 'stim' frame number is the onset of the tag
            onset_ms = 0
            onset_frame = 0

            # We got everything, add to table
            self._append(key, event_type, tag, onset_frame, stim, stim_session)


    def _make_tuples(self, key):
        # Insert into main table
        manual_insert_skip_duplicate(self, key)

        # Get number of frmes per recording
        frames_shift = (Sessions & key).fetch1('frames_shift')

        # Iterate over recordings
        hdf = (Experiment & key).fetch('hdf_path')
        for rec_n, h in enumerate(hdf):
            h = h.replace('.hdf5', '_forFede.hdf5')

            try:
                f, keys, subkeys, allkeys = open_hdf(h)
            except FileExistsError:
                print(f'File: {Path(h).name} does not exist!')

            # Get tag entries for each type of event
            for k in keys:
                if k in ('audio','visual', 'auditory', 'visual_audio'):
                    self.process_stim_evoked(k, f, frames_shift, key, rec_n)
                else:
                    self.process_other(k, f, frames_shift, key, rec_n)

