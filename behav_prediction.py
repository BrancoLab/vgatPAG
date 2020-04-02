
import sys
sys.path.append('./')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import time
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture as GMM
import cv2
import multiprocessing as mp

from fcutils.maths.utils import interpolate_nans, get_random_rows_from_array
from fcutils.plotting.colors import salmon
from fcutils.plotting.utils import clean_axes
from fcutils.video.utils import get_cap_from_file, get_cap_selected_frame, get_video_params, open_cvwriter

from vgatPAG.database.db_tables import *
from vgatPAG.paths import main_data_folder




class BehaviourPrediction:
    _umap_params = dict(
        n_neighbors=15,
        n_components=3,
        metric='euclidean',
        metric_kwds=None,
        n_epochs=None,
        learning_rate=1.0,
        init='spectral',
        min_dist=0.1,
        spread=1.0,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        transform_queue_size=4.0,
        a=None,
        b=None,
        random_state=None,
        angular_rp_forest=False,
        target_n_neighbors=-1,
        target_metric='categorical',
        target_metric_kwds=None,
        target_weight=0.5,
        transform_seed=42,
        verbose=False,
    )

    def __init__(self, n_behav_classes=8, umapper=None):
        self.tracking = None
        self.n_behav_classes = n_behav_classes
        self.umapper = umapper

    # ------------------------------------ Fit ----------------------------------- #
    def prep_recording_tracking_data(self, rec):
        self.rec = rec

        # Fetch data
        bp_tracking = pd.DataFrame((Trackings * Trackings.BodyPartTracking & f"rec_name='{rec}'" & "bp!='left_ear'" & "bp!='right_ear'").fetch())
        bs_tracking = pd.DataFrame((Trackings * Trackings.BodySegmentTracking & f"rec_name='{rec}'").fetch())

        # Isolate relevant
        speeds = np.vstack([s for s in bp_tracking.speed.values]).T
        orientations = np.vstack([s for s in bs_tracking.orientation.values]).T
        ang_vels = np.vstack([s for s in bs_tracking.angular_velocity.values]).T
        lengths = np.vstack([s for s in bs_tracking.bone_length.values]).T

        # Concatenate
        self.tracking = interpolate_nans(np.hstack([speeds, orientations, ang_vels, lengths]))

        print(f"Collated tracking, found {self.tracking.shape[1]} features for {self.tracking.shape[0]} frames")        
        return self.tracking



    def fit(self, max_n_frames=-1, plot=False,  **kwargs):
        if self.tracking is None: raise ValueError("Need to provide data first")

        # prep umap params
        umap_params = {p:kwargs.pop(p, v) for p,v in self._umap_params.items()}

        # Select subset of data to use
        if max_n_frames == -1:
            data= self.tracking.copy()
        else:
            data = get_random_rows_from_array(self.tracking, max_n_frames)

        # Apply umap
        print("Applying umap")
        start = time.time()
        umapper = umap.UMAP(**umap_params)
        embedding = umapper.fit_transform(self.tracking[:max_n_frames, :])
        self.umapper = umapper
        print("UMAP embedding took {} seconds".format(round(time.time() - start, 2)))

        # Apply GMM to embedded data
        print(f"Fitting GMM to extract {self.n_behav_classes} clusters")
        labels = GMM(n_components=self.n_behav_classes).fit_predict(embedding)

        # Visualise results
        if plot:
            self.visualise_embedding(embedding, labels)

        self.embedding = embedding
        return embedding

    # ------------------------------ Predicting data ----------------------------- #
    def predict_data(self, data, plot=False):
        start = time.time()
        print(f"Predicting behaviour on {data.shape[0]} frames, using {self.n_behav_classes} behaviour classes")
        embedding = self.umapper.transform(data)
        labels = GMM(n_components=self.n_behav_classes).fit_predict(embedding)
        print("Prediction took {} seconds".format(round(time.time() - start, 2)))

        # Visualise results
        if plot:
            self.visualise_embedding(embedding, labels)

        return embedding, labels

    # ------------------------------- Visualisation ------------------------------ #
    def visualise_embedding(self, embedding, labels):
        if embedding.shape[1] > 3: 
            print("cannot plot wiht >3 components, skipping")
            return

        elif embedding.shape[1] == 3:
            f = plt.figure()
            ax = f.add_subplot(111, projection='3d')
            ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=labels, cmap='tab20')

        else:
            f, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab20')
            ax.set(title='UMAP embedding of tracking features')
        clean_axes(f)

    def make_behav_class_clip(self, args):
        behav_label, labels, savepath = args

        print(f"Saving example clip for beahviour class {behav_label}")

        # Get video file
        recdata = (Recording & f"rec_name='{self.rec}'").fetch(as_dict=True)[0]
        videopath = os.path.join(main_data_folder, recdata['mouse'], recdata['sess_name'], recdata['videofile'])

        # Prep CV2 writer
        cap = get_cap_from_file(videopath)
        nframes, width, height, fps = get_video_params(cap)
        writer = open_cvwriter(savepath, w=width, h=height, framerate=fps, iscolor=True)

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (50, 50)
        fontScale              = 1
        lineType               = 2
        text_color             = (30, 220, 30)

        # Make frames
        for i in range(50000):
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
        print(f"    finished saving example clip for beahviour class {behav_label}")

    def make_all_behav_clips(self, labels):
        classes = list(set(labels))

        pool = mp.Pool(mp.cpu_count()-4)
        pool.map(self.make_behav_class_clip, [(behav_class, labels, f"test_class_{i}.mp4") for i, behav_class in enumerate(classes)])
        pool.close()



if __name__ == "__main__":
    # Select an example session for testing
    rec = Recording().fetch("rec_name")[0]

    bp = BehaviourPrediction()

    tracking = bp.prep_recording_tracking_data(rec)

    
    # Embedd features and plot
    embedding  = bp.fit(max_n_frames=10000, plot=True,  n_components=3)

    # Predict data    
    embed, labels = bp.predict_data(tracking, plot=False)

    # Make videoclips
    bp.make_all_behav_clips(labels)


    plt.show()

    




