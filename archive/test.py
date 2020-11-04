# %%
import matplotlib.pyplot as plt
import numpy as np

from fcutils.maths.utils import derivative

from Analysis  import (
    mouse_sessions,
    sessions, 
    get_mouse_session_data
)
from rich.progress import track

# %%
for mouse, sess, sessname in track(mouse_sessions):
    tracking, ang_vel, speed, shelter_distance, signals, nrois, is_rec = get_mouse_session_data(mouse, sess, sessions)
    if is_rec is None:
        continue

    f, axarr = plt.subplots(nrows=8, figsize=(10, 16))
    f.suptitle(sessname)

    start_ends = derivative(is_rec)
    is_rec[start_ends == 1] = 0  # remove first rec frame to remove artifacts
    is_rec[start_ends == -1] = 0  # remove first rec frame to remove artifacts

    e = np.where(start_ends == -1)[0][0]

    for n, ax in enumerate(axarr):
        sig = signals[n][is_rec == 1]
      
        ax.plot(sig)
        # ax.axhline(perc, lw=2, color='r')

        # if perc < .1:
        #     raise ValueError

    # break
# %%
