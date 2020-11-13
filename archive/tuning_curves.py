# %%
import numpy as np
import matplotlib.pyplot as plt
from rich.progress import track
import pandas as pd

from Analysis  import (
        get_mouse_session_data,
        mouse_sessions,
        sessions,
)
import seaborn as sns

from Analysis.misc import get_tiff_starts_ends, get_chunked_dff
from fcutils.plotting.utils import save_figure, clean_axes
from fcutils.plotting.plot_elements import plot_mean_and_error
from fcutils.maths.utils import derivative

# %%
"""
    For each ROI, show tuning curve relative to different variables
"""

for mouse, sess, sessname in track(mouse_sessions, description='plotting tuning curves'):
    
    # Get data
    tracking, ang_vel, speed, shelter_distance, dffs, signals, nrois, is_rec, roi_ids = \
                        get_mouse_session_data(mouse, sess, sessions, hanning_window=6)
    tiff_starts, tiff_ends = get_tiff_starts_ends(is_rec)

    is_rec[speed < .2] = 0

    vrs = (
        # ('x pos', 'salmon', tracking.x.values[is_rec==1]),
        # ('x speed', 'teal', derivative(tracking.x.values)[is_rec==1]),
        ('speed', 'skyblue', speed[is_rec==1]),
        # ('y speed', 'orange', derivative(tracking.y.values)[is_rec==1]),
    )

    for n, (sig, rid) in enumerate(zip(signals, roi_ids)):
        # get chunked dff
        dff = get_chunked_dff(sig, tiff_starts, tiff_ends)[is_rec==1]

        # Plot tuning curve
        f, axarr = plt.subplots(ncols=2, nrows=2, figsize=(16, 9))
        axarr = axarr.flatten()

        for c, (name, col, var) in enumerate(vrs):
            df = pd.DataFrame(dict(dff=dff, x=var))
            binned = pd.cut(df['x'], bins=10)
            mean = df.groupby(binned).mean()
            std = df.groupby(binned).std()


            # axarr[c].scatter(var[::100], dff[::100], color=col, label=name, alpha=.3)
            sns.regplot(var[::100], dff[::100], color=col,label=name, truncate=True, ax=axarr[c], 
                        scatter_kws=dict(alpha=.01))
            axarr[c].set(xlim=[0, 15])
            # plot_mean_and_error(mean['dff'], std['dff'], axarr[c], x=mean['x'], color=col, label=name)
            # axarr[c].legend()
            axarr[c].set(ylabel='ROI DFF', xlabel=name, title=rid)
            clean_axes(f)
            f.tight_layout()
        # break
    break


# %%
mean
# %%
binned
# %%
df.head()
# %%
binned
# %%
df.groupby(binned).count()
# %%var
# %%
var
# %%
plt.plot(var)
# %%
