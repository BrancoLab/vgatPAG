# %%
import sys
sys.path.append('./')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from vgatPAG.database.db_tables import Tracking


# %%
Tracking().populate()

# %%
