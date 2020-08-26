# %%
import matplotlib.pyplot as plt
from brainrender.colors import makePalette, colorMap
import numpy as np

from palettable.cmocean.sequential import Matter_8 as CMAP

from fcutils.plotting.utils import calc_nrows_ncols, set_figure_subplots_aspect, clean_axes

from Analysis  import (
        mice,
        sessions,
        recordings,
        recording_names,
        stimuli,
        clean_stimuli,
        get_mouse_session_data,
        sessions_fps,
        mouse_sessions,
)
from Analysis.tag_aligned import (
    manual_tags,
    get_tags_by,
)

# %%
"""
    For each tag plot all ROIs signals aligned

    UTILS
"""

def make_signal_ax(ax, ttl, start, end):
    t = end - start
    ax.axvline(n_frames_pre, lw=2, color='k', ls='--', alpha=.3, zorder=-1)

    ax.set(title=ttl, xticks=[0, n_frames_pre, t], 
                    xticklabels=[-n_sec_pre, 0, n_sec_post],
                    xlabel='time (s)', ylabel='signal')

def plot_tracking_trace(ax, tracking, color, label, start, end, mark_frame = None):
    # plot tracking
    ax.plot(tracking.x[start:end], tracking.y[start:end], label=str(tag.stim_frame),
        color=color, lw=2)

    # mark start and end
    ax.scatter(tracking.x[start], tracking.y[start], s=150,
            color=color, lw=1, edgecolors=[.2, .2, .2], zorder=99)

    if mark_frame is not None:
        ax.scatter(tracking.x[mark_frame], tracking.y[mark_frame], s=150,
                color='white', lw=2, edgecolors=color, zorder=99)

def plot_signal_trace(ax, sig, color, start, end, mark_frame=None):
    t = end-start
    # plot tracking
    ax.plot(sig[start:end], label=str(tag.stim_frame),
        color=[.4, .4, .4], lw=6, zorder=-1)
    ax.plot(sig[start:end], label=str(tag.stim_frame),
        color=color, lw=4)

    if mark_frame is not None:
        ax.scatter(mark_frame-start, sig[mark_frame], s=150,
                color='white', lw=2, edgecolors=color, zorder=99)

def norm_sig(sig, start, end, n_sec_post, ):
    raise NotImplementedError('need to think about wht the best way to norm this stuff is')
    # baseline = np.mean(sig[start: start + n_frames_pre - 1])
    # return sig / baseline

def get_next_tag(frame, tags, max_delay = 1000):
    """
        Selects the first tag to happend after a given frame
        as long as not too long a delay happend
    """
    nxt = tags.loc[(tags.session_frame - frame > 0)&(tags.session_frame - frame < 1000)]
    if not len(nxt): return None
    else: 
        return nxt.session_frame.values[0]



# %%

# TODO remove trials when the mouse gets caught in the arena

# TODO make manual shelter location annotattion stuff
# TODO look into out of bounds tags

# TODO remove artifacts when the mouse gets in the shelter

_colors = []

# Params
n_sec_pre = 2
n_frames_pre = n_sec_pre * 30
n_sec_post = 6
tag_type = 'VideoTag_B'

NORMALIZE = False # if true divide sig by the mean of the sig in n_sec_pre

event_types = list(manual_tags.event_type.unique())

raise ValueError('The tags for  18JUN19 not working')

for mouse, sess, sessname in mouse_sessions:
    print(f'\nProcessing {sessname}')
    # Get data
    tracking, ang_vel, speed, shelter_distance, signals, nrois, is_rec = get_mouse_session_data(mouse, sess, sessions)
    tags = get_tags_by(mouse=mouse, sess_name=sess, event_type='Loom_Escape', tag_type=tag_type)

    if not len(tags):
        print(f'Didnt find any tags of tupe {tag_type} in session {sessname}')

    at_shelt_tags = get_tags_by(mouse=mouse, sess_name=sess, event_type='Loom_Escape', tag_type='VideoTag_C')
    
    # Prep figure
    rows, cols = calc_nrows_ncols(len(signals) + 1, (30, 15))
    f, axarr = plt.subplots(ncols=cols, nrows=rows, figsize = (30, 15))
    f.suptitle(f'{sessname} traces aligned to {tag_type}')
    axarr = axarr.flatten()

    colors = [colorMap(i, name=CMAP.mpl_colormap, vmin=-1, vmax=len(tags)+1) for i in range(len(tags))]

    # Plot data
    for color, (i, tag) in zip(colors, tags.iterrows()):
        start = tag.session_frame - 30 * 2
        end = tag.session_frame + 30 * 2
        nxt_at_shelt = get_next_tag(start, at_shelt_tags)

        if start > len(tracking):
            print(f'!! tag frame {start} is out of bounds: # frames: {len(tracking)}')
            raise ValueError
            continue

        plot_tracking_trace(axarr[0], tracking, color, f'frame {start}', start, end, mark_frame=nxt_at_shelt)

        
        for roin, sig in enumerate(signals):
            plot_signal_trace(axarr[roin+1], 
                                sig if not NORMALIZE else norm_sig(sig, start, end, n_sec_pre), 
                                color, start, end,
                                mark_frame=nxt_at_shelt)

    # fig figure
    for n, ax in enumerate(axarr[1:]):
        if n >= len(signals):
            ax.axis('off')
        else:
            make_signal_ax(ax, f'ROI {n}', start, end)

    axarr[0].set(title='tracking', xlabel='X', ylabel='Y')

    clean_axes(f)
    set_figure_subplots_aspect(right=.85, top=.9, wspace=.4)

    # break

# %%
