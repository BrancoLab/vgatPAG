
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
from scipy.signal import resample
from sklearn import preprocessing

from fcutils.maths.utils import interpolate_nans, get_random_rows_from_array, rolling_mean
from fcutils.maths.geometry import calc_ang_velocity
from fcutils.plotting.colors import salmon
from fcutils.plotting.colors import colorMap
from fcutils.file_io.io import load_pickle, save_pickle
from fcutils.maths.utils import derivative
from fcutils.plotting.utils import clean_axes
from fcutils.video.utils import get_cap_from_file, get_cap_selected_frame, get_video_params, open_cvwriter

# from behaviour.tracking.tracking import compute_bone_semgent

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

    def __init__(self, n_behav_classes=4, umapper=None, fps=30):
        self.features = None
        self.n_behav_classes = n_behav_classes
        self.umapper = umapper
        self.fps = fps

    # ------------------------------ Preprocess data ----------------------------- #
    def prepare_features(self, recs):
        self.rec = recs[0]

        features = []
        for n, rec in enumerate(recs):
            # Fetch data
            bp_tracking = pd.DataFrame((Trackings * Trackings.BodyPartTracking & f"rec_name='{rec}'" & "bp!='left_ear'" & "bp!='right_ear'").fetch())
            bs_tracking = pd.DataFrame((Trackings * Trackings.BodySegmentTracking & f"rec_name='{rec}'").fetch())

            snout = bp_tracking.loc[bp_tracking.bp == "snout"].iloc[0]
            body = bp_tracking.loc[bp_tracking.bp == "body"].iloc[0]
            neck = bp_tracking.loc[bp_tracking.bp == "neck"].iloc[0]
            tail_base = bp_tracking.loc[bp_tracking.bp == "tail_base"].iloc[0]

            whole_body_segment = bs_tracking.loc[(bs_tracking.bp1 == "snout") & (bs_tracking.bp2 == "tail_base")].iloc[0]
            head_segment = bs_tracking.loc[(bs_tracking.bp1 == "snout") & (bs_tracking.bp2 == "neck")].iloc[0]

            # Create features from tracking
            rec_features = []
            rec_features.append(whole_body_segment.bone_length)
            rec_features.append(whole_body_segment.bone_length - head_segment.bone_length)
            rec_features.append(snout.speed)
            rec_features.append(tail_base.speed)
            rec_features.append(derivative(whole_body_segment.orientation))
            
            features.append(interpolate_nans(np.vstack(rec_features).T))

            if n == 0:
                first_tracking = features[0]

        # Put it all together 
        self.features = np.concatenate(features)

        # smooth data with a ~60ms window (2 frames at 30fps)
        window_size = int(np.ceil(60/(1000/self.fps)))
        smoothed = np.zeros_like(self.features)
        for i in range(self.features.shape[1]):
            smoothed[:, i] = rolling_mean(self.features[:, i], 2)
        self.features = smoothed

        # downasmple data to be at ~10 fps
        n_samples = smoothed.shape[0]
        n_samples_at_10_fps = int(np.ceil((n_samples / self.fps) * 10))
        self.features = resample(self.features, n_samples_at_10_fps)



        cprint(f"\nCollated tracking, found {self.features.shape[1]} features for {self.features.shape[0]} frames", "green", attrs=['bold'])        
        return self.features, first_tracking 
    # ------------------------------------ Fit ----------------------------------- #
    @staticmethod
    def fit_glm(embedding, n_components):
        cprint(f"Fitting GMM to extract {n_components} clusters",  "green", attrs=['bold'])
        labels = GMM(n_components=n_components).fit_predict(embedding)
        return labels


    def fit(self, max_n_frames=-1, plot=False,  **kwargs):
        if self.features is None: raise ValueError("Need to provide data first")

        # prep umap params
        umap_params = {p:kwargs.pop(p, v) for p,v in self._umap_params.items()}

        # Select subset of data to use
        if max_n_frames == -1 or max_n_frames >= self.features.shape[0]:
            data= self.features.copy()
        else:
            data = get_random_rows_from_array(self.features, max_n_frames)

        # Apply umap
        cprint(f"\nFitting UMAP with {data.shape[0]} frames - " + 
                    f"[{umap_params['n_neighbors']} knn, {umap_params['n_components']} components]",  
                    "green", attrs=['bold'])
        start = time.time()
        umapper = umap.UMAP(**umap_params)
        embedding = umapper.fit_transform(self.features[:max_n_frames, :])
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
            elif yn.lower() == "z":
                sys.exit()

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
    recs = Recording().fetch("rec_name")[:8]

    bp = BehaviourPrediction()
    features, first_tracking = bp.prepare_features(recs)

    if LOAD:
        bp.load()
    else:
        # Embedd features and plot
        embedding  = bp.fit(max_n_frames=40000, 
                plot=True,  
                n_neighbors = 25,)
        bp.save()

    # Predict data    
    embed, labels = bp.predict_data(first_tracking, plot=False)

    # Make videoclips
    bp.make_all_behav_clips(labels)


    plt.show()

    




