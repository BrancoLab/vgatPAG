# %%
import sys
sys.path.append('./')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


from fcutils.file_io.io import open_hdf
from fcutils.file_io.utils import check_create_folder

from vgatPAG.paths import summary_file



# %%
path = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Project_vgatPAG/analysis/doric/BF161p1/19JUN03/19JUN03_BF161p1_v1_tagAll.hdf5"
f, keys, subkeys, all_keys = open_hdf(path)

# %%
