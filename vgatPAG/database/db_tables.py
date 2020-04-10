import datajoint as dj
import pandas as pd
import datetime
import numpy as np
import os
from scipy.signal import medfilt as median_filter



from behaviour.tracking.tracking import prepare_tracking_data, compute_body_segments
from behaviour.utilities.signals import get_times_signal_high_and_low

from fcutils.file_io.io import open_hdf, load_yaml
from fcutils.file_io.utils import listdir, get_subdirs, get_last_dir_in_path, get_file_name
from fcutils.video.utils import get_cap_from_file, get_video_params
from fcutils.maths.utils import derivative

from vgatPAG.paths import summary_file, metadatafile, mice_folders
from vgatPAG.database.dj_config import start_connection, dbname, manual_insert_skip_duplicate
from vgatPAG.variables import miniscope_fps, tc_colors, shelter_width_px
from vgatPAG.utils import get_spont_homings, get_spont_out_runs

schema = start_connection()
np.warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------- #
#                                     UTILS                                    #
# ---------------------------------------------------------------------------- #

def get_trial_content(trial_class, session, trial_name):
    f, keys, subkeys, allkeys = open_hdf(summary_file)
    return dict(f['all'][trial_class][session][trial_name])

def get_mouse_session_subdirs(mouse):
        folders = [fld for fld in mice_folders if mouse in fld]
        if len(folders) != 1:
            raise ValueError("smth went wrong")
        return get_subdirs(folders[0])

def get_session_folder(mouse=None, sess_name=None, **kwargs):
    mouse_flds = get_mouse_session_subdirs(mouse)
    sess_fld = [fld for fld in mouse_flds if sess_name in fld]
    if len(sess_fld) != 1:
        raise ValueError("Something went wrong")
    
    return sess_fld[0]

# ---------------------------------------------------------------------------- #
#                                    TABLES                                    #
# ---------------------------------------------------------------------------- #

@schema
class Mouse(dj.Manual):
    definition = """
        mouse: varchar(64)
    """

    def populate(self):
        mice = load_yaml(metadatafile)['mice']
        for mouse in mice:
            manual_insert_skip_duplicate(self, {'mouse':mouse})

@schema
class Session(dj.Imported):
    definition = """
        -> Mouse
        sess_name: varchar(128)
    """

    def _make_tuples(self, key):
        # Get mouse folder
        subdirs = get_mouse_session_subdirs(key['mouse'])
        for sdir in subdirs:
            skey = key.copy()
            skey['sess_name'] = get_last_dir_in_path(sdir)
            manual_insert_skip_duplicate(self, skey)

@schema
class Recording(dj.Imported): #  In each session there might be multiple recordings...
    min_time_between_stims = 12 # seconds


    definition = """
        -> Session
        rec_name: varchar(128)
        ---
        rec_n: int
        videofile: varchar(256)
        aifile: varchar(256)
        n_frames: int #  number of frames in the behaviour video
        fps_behav: int # fps of behaviour video
        n_samples: int #  in AI file
    """
    def _make_tuples(self, key):
        session_fld = get_session_folder(**key)
        recs = load_yaml(metadatafile)['sessions'][key['mouse']][key['sess_name']]

        for n, rec in enumerate(sorted(recs)):
            # Get files
            rec_files = [f for f in os.listdir(session_fld) if rec in f]

            videos = [f for f in rec_files if f.endswith(".mp4") or f.endswith(".avi")]
            if len(videos) != 1: 
                if len(set([get_file_name(f) for f in videos])) == 1:
                    video = get_file_name(videos[0])+".mp4"
                else:
                    raise ValueError
            else: video = videos[0]

            ais = [fl for fl in rec_files if fl == f"{rec}.hdf5"]
            if len(ais) != 1: raise ValueError
            else: ai = ais[0]    

            # Open video and get number of frames
            nframes, width, height, fps = get_video_params(get_cap_from_file(os.path.join(session_fld, video)))

            # Open AI file and get number of samples
            f, keys, subkeys, allkeys = open_hdf(os.path.join(session_fld, ai))
            n_samples = len(f['AI']['0'][()])

            rkey= key.copy()
            rkey['rec_name'] = rec
            rkey['rec_n'] = n
            rkey['videofile'] = video
            rkey['aifile'] = ai
            rkey['n_frames'] = nframes
            rkey['fps_behav'] = fps
            rkey['n_samples'] = n_samples
            manual_insert_skip_duplicate(self, rkey)


    def get_recording_fps(self, **kwargs):
        return (self & kwargs).fetch1("fps_behav")

    def get_recording_stimuli_clean(self, **kwargs):
        # kwargs should have info about: sess_name, mouse, rec_name
        # Get recording's fps
        fps = self.get_recording_fps(**kwargs)
        min_frames_betweeen_stimuli = self.min_time_between_stims * fps

        # Fetch recording's stimuli and tracking data
        # Get stimuli
        vstims = (VisualStimuli & kwargs).fetch("frame")
        astims = (AudioStimuli & kwargs).fetch("frame")

        # Keep only stimuli that didn't happen immediately one after the other
        if np.any(vstims):
            vstims = [vstims[0]] + [s for n,s in enumerate(vstims[1:]) if s-vstims[n] > min_frames_betweeen_stimuli+1]
        if np.any(astims):
            astims = [astims[0]] + [s for n,s in enumerate(astims[1:]) if s-astims[n] > min_frames_betweeen_stimuli+1]
        astims = [s for s in astims if s not in vstims] # for some reason some are repeated

        return vstims, astims



@schema
class VisualStimuli(dj.Imported):
    definition = """
        -> Session
        rec_name: varchar(128)
        frame: int
        ---
        protocol_name: varchar(256)

    """
    def _make_tuples(self, key):
        session_fld = get_session_folder(**key)
        recs = load_yaml(metadatafile)['sessions'][key['mouse']][key['sess_name']]

        # Get each session's recordin AI file
        for n, rec in enumerate(sorted(recs)):
            rec_files = [f for f in os.listdir(session_fld) if rec in f]

            ais = [fl for fl in rec_files if fl == f"{rec}.hdf5"]
            if len(ais) != 1: raise ValueError
            else: ai = ais[0]  
            f, keys, subkeys, allkeys = open_hdf(os.path.join(session_fld, ai))

            # Get stim start and protocol name
            try:
                protocol_names = dict(f['Visual Stimulation'])['Visual Protocol Names'][()].decode("utf-8") 
            except KeyError as err:
                return 

            protocol_names = protocol_names.replace("[", "").replace("]", "").replace("'", "")
            protocol_names = protocol_names.split(", u")

            start_frames = np.where(dict(f['Visual Stimulation'])['Visual Stimulus Start Indices'][()] > 0)[0]

            if len(start_frames) != len(protocol_names): raise ValueError

            for sf, pc in zip(start_frames, protocol_names):
                skey = key.copy()
                skey['frame'] = sf
                skey['protocol_name'] = pc
                skey['rec_name'] = rec
                manual_insert_skip_duplicate(self, skey)




@schema
class AudioStimuli(dj.Imported):
    definition = """
        -> Session
        rec_name: varchar(128)
        frame: int
        ---
        protocol_name: varchar(256)

    """
    def _make_tuples(self, key):
        session_fld = get_session_folder(**key)
        recs = load_yaml(metadatafile)['sessions'][key['mouse']][key['sess_name']]

        # Get each session's recordin AI file
        for n, rec in enumerate(sorted(recs)):
            rec_files = [f for f in os.listdir(session_fld) if rec in f]

            ais = [fl for fl in rec_files if fl == f"{rec}.hdf5"]
            if len(ais) != 1: raise ValueError
            else: ai = ais[0]  
            f, keys, subkeys, allkeys = open_hdf(os.path.join(session_fld, ai))

            # Get stim start and protocol name
            try:
                protocol_names = dict(f['Audio Stimulation'])['Audio Stimuli Names'][()].decode("utf-8") 
            except KeyError as err:
                return 

            protocol_names = protocol_names.replace("[", "").replace("]", "").replace("'", "")
            protocol_names = protocol_names.split(", u")
            protocol_names =[pc.split("\\\\")[-1] for pc in protocol_names]

            start_frames = np.where(dict(f['Audio Stimulation'])['Audio Stimuli Start Indices'][()] > 0)[0]

            if len(start_frames) != len(protocol_names): raise ValueError

            for sf, pc in zip(start_frames, protocol_names):
                skey = key.copy()
                skey['frame'] = sf
                skey['protocol_name'] = pc
                skey['rec_name'] = rec
                manual_insert_skip_duplicate(self, skey)

@schema
class Trackings(dj.Imported):
    definition = """
        -> Recording
    """
    bparts = ['snout', 'left_ear', 'right_ear', 'neck', 'body', 'tail_base']
    bsegments = {'head':('snout', 'neck'), 
                'upper_body':('neck', 'body'),
                'lower_body':('body', 'tail_base'),
                'whole_body':('snout', 'tail_base')}

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

    class BodySegmentTracking(dj.Part):
        definition = """
            -> Trackings
            bp1: varchar(64)
            bp2: varchar(64)
            ---
            orientation: longblob
            angular_velocity: longblob
            bone_length: longblob
        """

    # Populate method
    def _make_tuples(self, key):
        session_fld = get_session_folder(**key)
        recs = load_yaml(metadatafile)['sessions'][key['mouse']][key['sess_name']]

        for n, rec in enumerate(sorted(recs)):
            if rec != key['rec_name']: continue
            
            # Get files
            rec_files = [f for f in os.listdir(session_fld) if rec in f]
            tracking_file = [f for f in rec_files if 'dlc' in f.lower() and f.endswith('.h5')]
            if len(tracking_file ) != 1: raise ValueError
            else: tracking_file  = os.path.join(session_fld, tracking_file[0])

            # Insert entry into main class
            self.insert1(key)

            # Load and clean tracking data
            bp_tracking = prepare_tracking_data(tracking_file, median_filter=True,
                                            likelihood_th=0.9,
            								fisheye=False, common_coord=False, compute=True)
            bones_tracking = compute_body_segments(bp_tracking, self.bsegments)

            # Insert into the bodyparts tracking
            for bp, tracking in bp_tracking.items():
            	bp_key = key.copy()
            	bp_key['bp'] = bp
                
            	bp_key['x'] = tracking.x.values
            	bp_key['y'] = tracking.y.values
            	bp_key['speed'] = tracking.speed.values
            	bp_key['dir_of_mvmt'] = tracking.direction_of_movement.values
            	bp_key['angular_velocity'] = tracking.angular_velocity.values

            	self.BodyPartTracking.insert1(bp_key)

            # Insert into the body segment data
            for bone, (bp1, bp2) in self.bsegments.items():
            	segment_key = key.copy()
            	segment_key['bp1'] = bp1
            	segment_key['bp2'] = bp2

            	segment_key['orientation'] = bones_tracking[bone]['orientation'].values
            	segment_key['angular_velocity'] = bones_tracking[bone]['angular_velocity'].values
            	segment_key['bone_length'] = bones_tracking[bone]['bone_length'].values

            	self.BodySegmentTracking.insert1(segment_key)


    def get_recording_tracking_clean(self, **kwargs):
        # kwargs should have info about: sess_name, mouse, rec_name
        body_tracking = np.vstack((self * self.BodyPartTracking & kwargs & "bp='body'").fetch1("x", "y", "speed"))

        # Get an average angular velocity  o minimise artifacts
        ang_vel1 = median_filter((self * self.BodySegmentTracking & kwargs & "bp1='neck'" & "bp2='body'").fetch1("angular_velocity"), kernel_size=11)
        ang_vel2 = median_filter((self * self.BodySegmentTracking & kwargs & "bp1='body'" & "bp2='tail_base'").fetch1("angular_velocity"), kernel_size=11)

        ang_vel = np.median(np.vstack([ang_vel1, ang_vel2]), axis=0)
        speed = median_filter(body_tracking[2, :], kernel_size=11)
        shelter_distance = body_tracking[0, :]-shelter_width_px-200

        return body_tracking, ang_vel, speed, shelter_distance

@schema
class TiffTimes(dj.Imported):
    definition = """
        -> Recording
        ---
        is_ca_recording: longblob    # 1 when Ca recording on, 0 otherwise
        starts: longblob
        ends: longblob
    """
    def _make_tuples(self, key):
        session_fld = get_session_folder(**key)

        fl = [f for f in listdir(session_fld) if 'tifflengths.npy' in f][0]
        start_times = np.load(fl)

        aifile = (Recording & key).fetch1("aifile")
        f, keys, subkeys, allkeys = open_hdf(os.path.join(session_fld, aifile))

        roi_data = f['CRaw_ROI_1'][()]
        signal = np.ones_like(roi_data)
        signal[derivative(roi_data )== 0] = 0

        starts, ends = get_times_signal_high_and_low(signal, th=.5)
        if len(starts) != len(ends) or starts[0] > ends[0]: raise ValueError
        
        key['is_ca_recording'] = signal
        key['starts'] = starts
        key['ends'] = ends
        manual_insert_skip_duplicate(self, key)




@schema
class Roi(dj.Imported):
    definition = """
        -> Recording
        roi_id: varchar(128)
        ---
        signal: longblob  # Concatenated signal with nans where no recording was happening

    """
    def _make_tuples(self, key):
        session_fld = get_session_folder(**key)
        
        aifile = (Recording & key).fetch1("aifile")
        f, keys, subkeys, allkeys = open_hdf(os.path.join(session_fld, aifile))

        rois = [k for k in keys if 'CRaw_ROI' in k]
        print(f"{key['rec_name']} -> {len(rois)} rois")

        for roi in rois:
            roi_trace = f[roi][()] 
            
            rkey = key.copy()
            rkey['roi_id'] = roi
            rkey['signal'] = roi_trace

            manual_insert_skip_duplicate(self, rkey)

    def get_roi_signal_clean(self, recording_name, roi_id): 
        # Returns a roi's trace but only for the frames where recording was on
        is_recording = (TiffTimes & f"rec_name='{recording_name}'").fetch1("is_ca_recording")
        signal = (self & f"rec_name='{recording_name}'" & f"roi_id='{roi_id}'").fetch1("signal")
        return np.array(signal)[np.where(is_recording)]

    def get_roi_signal_at_random_time(self, recording_name, roi_id, frames_pre, frames_post):
        # Returns a section of roi signal cut out at a random time
        signal = self.get_roi_signal_clean(recording_name, roi_id)
        cut = frames_pre + frames_post + 2
        start = np.random.randint(cut, len(signal)-cut)  # cut out initial and last frames to make sure it has correct length
        return signal[start-frames_pre:start+frames_post]

    def get_recordings_rois(self, **kwargs):
        roi_ids, roi_sigs = (self & kwargs).fetch("roi_id", "signal", as_dict=False)
        nrois = len(roi_ids)
        return roi_ids, roi_sigs, nrois

# ---------------------------------------------------------------------------- #
#                                    EVENTS                                    #
# ---------------------------------------------------------------------------- #
# Stuff like trial start, escape onset, arrival at shelter, spontaneous escapes etc.

@schema
class Event(dj.Imported):
    max_homing_duration = 4 # seconds
    max_run_duration = 4 # second
    min_time_after_stim = 6 # seconds
    spont_initiation_speed = 4
    escape_initiation_speed = 6

    max_event_duration = 12 # seconds


    definition = """
        -> Recording
    """

    class Evoked(dj.Part): # stim evoked events
        definition = """
            -> Event
            frame: int
            type: enum("stim_onset", "escape_onset", "escape_peak_speed", "shelter_arrival")
            --- 
            stim_type: enum("visual", "audio")
        """
    
    class Spontaneous(dj.Part): # stim evoked events
        definition = """
            -> Event
            frame: int
            type: enum("homing", "homing_peak_speed", "outrun")
        """
    
    def _make_tuples(self, key):
        # Prepr some vars
        fps = Recording().get_recording_fps(**key)
        max_homing_duration = self.max_homing_duration * fps
        max_run_duration = self.max_run_duration * fps
        min_time_after_stim = self.min_time_after_stim * fps

        max_event_duration = self.max_event_duration * fps

        # Get stimuli
        vstims, astims = Recording().get_recording_stimuli_clean(**key)

        # Get tracking
        body_tracking, ang_vel, speed, shelter_distance = Trackings().get_recording_tracking_clean(**key)

        # Get spontaneous homings and runs from tracking
        homings = get_spont_homings(shelter_distance, speed, astims, vstims, max_homing_duration, min_time_after_stim, self.spont_initiation_speed)
        runs = get_spont_out_runs(shelter_distance, speed, astims, vstims, max_run_duration, min_time_after_stim, self.spont_initiation_speed)

        # now it's time to populate stuff
        manual_insert_skip_duplicate(self, key) # insert into main tabel

        # Insert into evoked subtable
        for stims, stims_type in zip([astims, vstims], ["audio", "visual"]):
            for stim in stims:
                # Get the escape onset from the speed data
                try:
                    estart = np.where(speed[stim+5:stim+max_event_duration]>=escape_initiation_speed)[0][0]+stim+5
                except:
                    continue # if there's no escape ignore
                
                # Get if/when the mouse reacehd the shelter and ignore otherwsie
                try:
                    at_shelter = np.where(shelter_distance[stim:stim+max_event_duration] <= 0)[0][0] + stim
                except:
                    continue

                # Get the time at which the peak of escape speed is reached
                speed_peak = np.argmax(np.nan_to_num(speed[stim:stim+max_event_duration])) + stim

                # Now add all of these events to the tables
                events = [stim, estart, at_shelter, speed_peak]
                classes = ["stim_onset", "escape_onset", "shelter_arrival", "escape_peak_speed"]
                for ev, cl in zip(events, classes):
                    ekey = key.copy()
                    ekey['frame'] = ev
                    ekey['type'] = cl
                    ekey['stim_type'] = stims_type
                    manual_insert_skip_duplicate(self.Evoked, ekey)

        # And now populate spontaneous subtables
        for spont in homings:
            onset = np.argmax(np.nan_to_num(speed[spont:spont+max_event_duration])) + spont

            for ev, cl in zip([spont, onset], ["homing", "homing_peak_speed"]):
                skey = key.copy()
                skey['frame'] = ev
                skey['type'] = cl
                manual_insert_skip_duplicate(self.Spontaneous, skey)

        for spont in runs:
            skey = key.copy()
            skey['frame'] = spont
            skey['type'] = "outrun"
            manual_insert_skip_duplicate(self.Spontaneous, skey)



# ---------------------------------------------------------------------------- #
#                    MANUALLY DEFINE STUFF FROM SUMMARY HDF                    #
# ---------------------------------------------------------------------------- #

@schema 
class ManualTrials(dj.Manual):
    definition = """
        trial_class: varchar(128)
        -> Mouse
        -> Session
        trial_name: varchar(128)
        ---
        frame: int
        manual_sess_name: varchar(128)

    """

    def populate(self):
        f, keys, subkeys, allkeys = open_hdf(summary_file)

        for trial_class in subkeys['all']:
            sessions = list(dict(f['all'][trial_class]).keys())
            for session in sessions:
                trials = list(dict(f['all'][trial_class][session]).keys())
                for trial in trials:
                    tkey = dict(
                        trial_class = trial_class,
                        mouse = session.split("_")[1],
                        sess_name = session.split("_")[2],
                        frame = int(trial.split("_")[1]),
                        trial_name = trial,
                        manual_sess_name = session,
                    )
                    manual_insert_skip_duplicate(self, tkey)

    def get_trial_classes(self):
        tcs = list(set(self.fetch("trial_class")))
        self.trial_classes = tcs
        return tcs

@schema
class ManualROIs(dj.Imported):
    definition = """
        -> ManualTrials
        roi_id: varchar(128)
        ---
        signal: longblob
    """

    parent = ManualTrials
    trial_classes = None

    def _make_tuples(self, key):
        tc = key['trial_class']
        session = (ManualTrials & key).fetch1("manual_sess_name")

        f, keys, subkeys, allkeys = open_hdf(summary_file)
        trial = dict(f['all'][tc][session][key['trial_name']])

        rois = [k for k in trial.keys() if 'CRaw_ROI' in k]
        for roi in rois:
            roi_trace = trial[roi][()]

            rkey = key.copy()
            rkey['roi_id'] = roi
            rkey['signal'] = roi_trace

            manual_insert_skip_duplicate(self, rkey)

    def get_mice(self):
        return list(set(self.fetch("mouse")))

    def get_mouse_sessions(self, mouse):
        mice = self.get_mice()
        if mouse not in mice: raise ValueError

        return list(set((self & f"mouse='{mouse}'").fetch("sess_name")))

    def get_session_trials(self, mouse, session):
        if not session in self.get_mouse_sessions(mouse):
            raise ValueError

        return pd.DataFrame((self & f"mouse='{mouse}'" & f"sess_name='{session}'").fetch())


    def get_trial_classes(self):
        self.trial_classes = self.parent().get_trial_classes()
        return self.trial_classes

    def get_trials_in_class(self, tclass):
        if self.trial_classes is None:
            self.get_trial_classes()

        if tclass not in self.trial_classes:
            raise ValueError("Invalid trial class")

        trials =  pd.DataFrame((self & f"trial_class='{tclass}'").fetch())
        mice = list(set(trials.mouse.values))
        sessions = {m:list(set(trials.loc[trials.mouse == m].sess_name.values)) for m in mice}
        trials_by_mouse = {m: trials.loc[trials.mouse == m] for m in mice}
        
        trials_by_session = {}
        for ss in sessions.values():
            for s in ss:
                trials_by_session[s] = trials.loc[trials.sess_name == s]

        return trials, mice, sessions, trials_by_mouse, trials_by_session