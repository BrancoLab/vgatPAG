
import sys
sys.path.append('./')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import umap.plot

import time
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture as GMM
import cv2
import multiprocessing as mp
from termcolor import cprint

from sklearn import preprocessing

from fcutils.maths.utils import interpolate_nans, get_random_rows_from_array
from fcutils.maths.geometry import calc_ang_velocity
from fcutils.plotting.colors import salmon
from fcutils.plotting.colors import colorMap
from fcutils.file_io.io import load_pickle, save_pickle

from fcutils.plotting.utils import clean_axes
from fcutils.video.utils import get_cap_from_file, get_cap_selected_frame, get_video_params, open_cvwriter

from vgatPAG.database.db_tables import *
from vgatPAG.paths import main_data_folder


from vtkplotter import show, Spheres
from vtkplotter import settings
settings.useDepthPeeling = True 
settings.useFXAA = True 


class BehaviourPrediction:
    _umap_params = dict(
        n_neighbors=15,  # higher values favour global vs local structure
        n_components=3,
        metric='euclidean',
        metric_kwds=None,
        n_epochs=None,
        learning_rate=1.0,
        init='spectral',
        min_dist=0.1, # min distance between point in low D embedding space. Low vals favour clustering
        spread=1.0,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        transform_queue_size=4.0,
        a=None,
        b=None,
        random_state=0,
        angular_rp_forest=False,
        target_n_neighbors=-1,
        target_metric='categorical',
        target_metric_kwds=None,
        target_weight=0.5,
        transform_seed=42,
        verbose=False,
    )

    def __init__(self, n_behav_classes=4, umapper=None):
        self.tracking = None
        self.n_behav_classes = n_behav_classes
        self.umapper = umapper

    # ------------------------------------ Fit ----------------------------------- #
    def prep_recording_tracking_data(self, recs):
        self.rec = recs[0]

        tracking = []
        for n, rec in enumerate(recs):
        # Fetch data
            bp_tracking = pd.DataFrame((Trackings * Trackings.BodyPartTracking & f"rec_name='{rec}'" & "bp!='left_ear'" & "bp!='right_ear'").fetch())
            bs_tracking = pd.DataFrame((Trackings * Trackings.BodySegmentTracking & f"rec_name='{rec}'").fetch())

            # Isolate relevant features
            speeds = np.vstack([s for s in bp_tracking.speed.values]).T
            ang_vels = np.vstack([s for s in bs_tracking.angular_velocity.values]).T
            lengths = np.vstack([s for s in bs_tracking.bone_length.values]).T

            # Compute angles between body segments
            head_body_angle = bs_tracking.loc[(bs_tracking.bp1 == 'snout')&(bs_tracking.bp2 == 'neck')].orientation.values[0] - \
                                bs_tracking.loc[(bs_tracking.bp1 == 'neck')&(bs_tracking.bp2 == 'body')].orientation.values[0]
            head_body_angle = head_body_angle.reshape(head_body_angle.shape[0], 1)

            body_tail_angle = bs_tracking.loc[(bs_tracking.bp1 == 'neck')&(bs_tracking.bp2 == 'body')].orientation.values[0] - \
                                bs_tracking.loc[(bs_tracking.bp1 == 'body')&(bs_tracking.bp2 == 'tail_base')].orientation.values[0]
            body_tail_angle = body_tail_angle.reshape(body_tail_angle.shape[0], 1)

            # Compute angular velocities
            # head_body_ang_vel = calc_ang_velocity(head_body_angle.ravel())
            # head_body_ang_vel = head_body_ang_vel.reshape(head_body_ang_vel.shape[0], 1)

            # body_tail_ang_vel = calc_ang_velocity(body_tail_angle.ravel())
            # body_tail_ang_vel = body_tail_ang_vel.reshape(body_tail_ang_vel.shape[0], 1)

            # Concatenate
            features = [speeds, ang_vels, lengths, head_body_angle, body_tail_angle] #, head_body_ang_vel, body_tail_ang_vel]
            tracking.append(interpolate_nans(np.hstack(features)))

            if n == 0:
                first_tracking = interpolate_nans(np.hstack(features))

        # Put it all together and normalise
        self.tracking = preprocessing.scale(np.vstack(tracking))
        # self.tracking = np.vstack(tracking)
        # self.tracking = self.tracking / np.linalg.norm(np.vstack(self.tracking), axis=0)

        cprint(f"\nCollated tracking, found {self.tracking.shape[1]} features for {self.tracking.shape[0]} frames", "green", attrs=['bold'])        
        return self.tracking, first_tracking

    @staticmethod
    def fit_glm(embedding, n_components):
        cprint(f"Fitting GMM to extract {n_components} clusters",  "green", attrs=['bold'])
        labels = GMM(n_components=n_components).fit_predict(embedding)
        return labels


    def fit(self, max_n_frames=-1, plot=False,  **kwargs):
        if self.tracking is None: raise ValueError("Need to provide data first")

        # prep umap params
        umap_params = {p:kwargs.pop(p, v) for p,v in self._umap_params.items()}

        # Select subset of data to use
        if max_n_frames == -1 or max_n_frames >= self.tracking.shape[0]:
            data= self.tracking.copy()
        else:
            data = get_random_rows_from_array(self.tracking, max_n_frames)

        # Apply umap
        cprint(f"\nFitting UMAP with {data.shape[0]} frames - " + 
                    f"[{umap_params['n_neighbors']} knn, {umap_params['n_components']} components]",  
                    "green", attrs=['bold'])
        start = time.time()
        umapper = umap.UMAP(**umap_params)
        embedding = umapper.fit_transform(self.tracking[:max_n_frames, :])
        self.umapper = umapper
        cprint("UMAP fitting took {} seconds".format(round(time.time() - start, 2)),  "green", attrs=['bold'])

        # Apply GMM to embedded data to select number of clusters
        yn = "n"
        while yn.lower() != "y":
            labels = self.fit_glm(embedding, self.n_behav_classes)
            self.visualise_embedding(embedding, labels, title="FIT outcome visuaisation")
            
            yn = input("\nAre you happy with the GMM clustering?    ")
            if yn.lower() == "n":
                self.n_behav_classes = int(input(f"Current number of clusters is {self.n_behav_classes}, what number should we use? "))
                print(f"set number of behaviors to {self.n_behav_classes}")

        self.embedding = embedding
        return embedding

    # ------------------------------ Predicting data ----------------------------- #
    def predict_data(self, data, plot=False):
        start = time.time()

        cprint(f"\nPredicting behaviour on {data.shape[0]} frames, using {self.n_behav_classes} behaviour classes",  "green", attrs=['bold'])

        embedding = self.umapper.transform(data)
        labels = GMM(n_components=self.n_behav_classes).fit_predict(embedding)

        cprint("Prediction took {} seconds".format(round(time.time() - start, 2)),  "green", attrs=['bold'])

        # Visualise results
        if plot:
            self.visualise_embedding(embedding, labels, title="Predicted frames")

        return embedding, labels

    # ------------------------------- Visualisation ------------------------------ #
    def visualise_embedding(self, embedding, labels, title=None):
        cprint("Visualising results of UMAP embedding [press q or esc to close]", color="green")
        if embedding.shape[1] > 3: 
            cprint("cannot plot wiht >3 components, skipping",  "red", attrs=['bold'])
            return

        elif embedding.shape[1] == 3:
            # Plot spehres with vtkplotter
            pts = [(embedding[i, 0], embedding[i, 1], embedding[i, 2]) for i in range(len(labels))]
            s0 = Spheres(pts, c=labels, r=.1, res=4, alpha=.7)  
            show(s0, axes=1, bg='white')
        else:
            umap.plot.points(self.umapper, labels=labels, theme='fire')

    def make_behav_class_clip(self, args):
        behav_label, labels, savepath = args

        cprint(f"Saving example clip for beahviour class {behav_label}",  "green", attrs=['bold'])

        # Get video file
        recdata = (Recording & f"rec_name='{self.rec}'").fetch(as_dict=True)[0]
        videopath = os.path.join(main_data_folder, recdata['mouse'], recdata['sess_name'], recdata['videofile'])

        # Prep CV2 writer
        cap = get_cap_from_file(videopath)
        nframes, width, height, fps, iscolor = get_video_params(cap)
        writer = open_cvwriter(savepath, w=width, h=height, framerate=fps, iscolor=True)

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (50, 50)
        fontScale              = 1
        lineType               = 2
        text_color             = (30, 220, 30)

        # Make frames
        for i in range(30000):
            ret, frame = cap.read()
            if not ret: break

            if labels[i] != behav_label: continue
            else: 
                cv2.putText(frame, f'Behaviour label {behav_label}', 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    text_color,
                    lineType)
                writer.write(frame)
        writer.release()
        cprint(f"    finished saving example clip for beahviour class {behav_label}",  "green", attrs=['bold'])

    def make_all_behav_clips(self, labels):
        cprint("Saving clips for inspection", color="green", attrs=['bold'])
        classes = list(set(labels))

        pool = mp.Pool(mp.cpu_count()-4)
        pool.map(self.make_behav_class_clip, [(behav_class, labels, f"test_class_{i}.mp4") for i, behav_class in enumerate(classes)])
        pool.close()

    # ------------------------------------ IO ------------------------------------ #
    def save(self):
        cprint("Saving fit UMAP model to file", color="green")
        save_pickle("behav_pred.pkl", self.umapper)
    
    def load(self):
        cprint("Loading fit UMAP model from file", color="green")
        self.umapper = load_pickle("behav_pred.pkl")











if __name__ == "__main__":
    LOAD = False
    # Select an example session for testing
    rec = Recording().fetch("rec_name")[:8]

    bp = BehaviourPrediction()
    tracking, first_tracking = bp.prep_recording_tracking_data(rec)

    if LOAD:
        bp.load()
    else:
        # Embedd features and plot
        embedding  = bp.fit(max_n_frames=50000, 
                plot=True,  
                n_neighbors = 500,)
        bp.save()

    # Predict data    
    embed, labels = bp.predict_data(first_tracking, plot=False)

    # Make videoclips
    bp.make_all_behav_clips(labels)


    plt.show()

    




