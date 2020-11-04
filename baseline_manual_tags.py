# %%
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from Analysis  import (
        mice,
        sessions,
        recordings,
        mouse_sessions,
        get_mouse_session_data,
)
from Analysis.tag_aligned import (
    manual_tags,
    get_tags_by,
    get_next_tag,
    get_last_tag,
)
from vgatPAG.database.db_tables import RoiDFF, TiffTimes

from fcutils.plotting.utils import save_figure, clean_axes
from pathlib import Path
import shutil
from vedo.colors import colorMap

fld = Path('D:\\Dropbox (UCL)\\Project_vgatPAG\\analysis\\doric\\Fede\\ddf_tag_aligned')
shutil.rmtree(fld)
fld.mkdir()
# %%
"""
    Plot raw ad dff traces aligned to a tag
    for all recordings and all rois

"""
tag_type = 'VideoTag_B'
event_type =  ['Loom_Escape', 'US_Escape', 'LoomUS_Escape'] #


fps = 30

n_sec_pre = 4 # rel escape onset
n_sec_post = 4 # # rel escape onset
n_frames_pre = n_sec_pre * fps
n_frames_post = n_sec_post * fps

for SMOOTH in (True, False):

    hanning = 6 if SMOOTH else None

    for mouse, sess, sessname in mouse_sessions:
        tracking, ang_vel, speed, shelter_distance, dffs, signals, nrois, is_rec, roi_ids = \
                            get_mouse_session_data(mouse, sess, sessions, hanning_window=hanning)
        tags = get_tags_by(mouse=mouse, sess_name=sess, event_type=event_type, tag_type=tag_type)

        tiff_starts, tiff_ends = (TiffTimes & f'mouse="{mouse}"' & f"sess_name='{sess}'").fetch("starts", "ends")
        last_ts = [0] + [t[-1] for t in tiff_starts][:-1]
        last_te = [0] + [t[-1] for t in tiff_ends][:-1]

        tiff_starts = np.concatenate([t+last_ts[n] for n,t in enumerate(tiff_starts)])
        tiff_ends = np.concatenate([t+last_te[n] for n,t in enumerate(tiff_ends)])

        f, axarr = plt.subplots(nrows=nrois, ncols=4,  figsize=(18, 4 * nrois), sharex=True)

        for n, (axl, axr, ax3, ax4, dff, sig, rid) in enumerate(zip(axarr[:, 0], axarr[:, 1], axarr[:, 2], axarr[:, 3], dffs, signals, roi_ids)):
            for count, (i, tag) in enumerate(tags.iterrows()):
                start = tag.session_frame - n_frames_pre
                end = tag.session_frame + n_frames_post

                bl_raw = np.mean(sig[start:tag.session_frame])
                bl_dff = np.mean(dff[start:tag.session_frame])

                c = colorMap(count, name='bwr', vmin=0, vmax=len(tags))
                axl.scatter(0, bl_raw, color=c, edgecolors='k', lw=1, s=150, zorder=99)
                axr.scatter(0, bl_dff, color=c, edgecolors='k', lw=1, s=150, zorder=99)

                axl.plot(sig[start:end], color='skyblue', lw=2, alpha=.6)
                axr.plot(dff[start:end], color='salmon', lw=2, alpha=.6)


                # baseline pre-tag
                th = np.percentile(sig[start:tag.session_frame], RoiDFF.DFF_PERCENTILE)
                ax3.plot((sig[start:end] - th)/th, color='seagreen', lw=2, alpha=.6)

                # baseline raw for whole doric chunk
                tf = [tf for tf in tiff_starts if tf < start][-1]
                te = [te for te in tiff_ends if te > end][0]

                if te<tf: raise ValueError

                th = np.percentile(sig[tf:te], RoiDFF.DFF_PERCENTILE)
                ax4.plot((sig[start:end] - th)/th, color='m', lw=2, alpha=.6)



            dffth = (RoiDFF & f"roi_id='{rid}'" & f"sess_name='{sess}'" & f"mouse='{mouse}'").fetch1('dff_th')
            axl.axhline(dffth, lw=4, color='k')

            axl.axvline(n_frames_pre, lw=2, color='r')
            axr.axvline(n_frames_pre, lw=2, color='r')
            ax3.axvline(n_frames_pre, lw=2, color='r')
            ax4.axvline(n_frames_pre, lw=2, color='r')

            title = rid + f' {count} traces'
            if n == 0: title = f'Tag: {tag_type} ' + title

            axr.set(title=title, ylabel='Delta F / F')
            axl.set(title=title, ylabel='raw')
            ax3.set(title=title + 'trial baseline', ylabel='raw trial baseline')
            ax3.set(title=title + 'doric chunk baseline', ylabel='raw doric chunk baseline')

        axl.set(xlabel='frames')
        axr.set(xlabel='frames')

        clean_axes(f)
        f.tight_layout()
        save_figure(f, fld / (sessname + f'_smooth_{SMOOTH}'))
