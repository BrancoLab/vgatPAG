# %%
import sys
sys.path.append('./')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from vgatPAG.bsoid.utils import boxcar_center


# %%
# Get just the XY tracking for each body part

tracking_file = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Project_vgatPAG/analysis/doric/BF161p1/19JUN03/19JUN03_BF161p1_v1-cam1DLC_resnet101_vgatPAGFeb25shuffle1_500000.h5"
tracking = pd.read_hdf(tracking_file)

tracking = tracking.values
tracking = np.delete(tracking, list(range(0, tracking.shape[1], 3)), axis=1)
tracking = np.delete(tracking, [2, 3, 4, 5], axis=1) # remove ears data
tracking.shape

# %%
