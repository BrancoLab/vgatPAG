# %%
import matplotlib.pyplot as plt
from brainrender.colors import makePalette, colorMap
import numpy as np
from scipy.signal import medfilt
from pathlib import Path
from palettable.cmocean.sequential import Matter_8 as CMAP
from palettable.cmocean.sequential import Speed_8 as CMAP2
from matplotlib.lines import Line2D

from fcutils.plotting.utils import calc_nrows_ncols, set_figure_subplots_aspect, clean_axes, save_figure

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
        pxperframe_to_cmpersec,
)
from Analysis.tag_aligned import (
    manual_tags,
    get_tags_by,
)

# %%
fld = Path('D:\\Dropbox (UCL - SWC)\\Project_vgatPAG\\analysis\\doric\\Fede\\plots\\ManualTagsAligned')

# %%
"""
    For each tag plot all ROIs signals aligned

    UTILS
"""

# ----------------------------------- Utils ---------------------------------- #

def make_signal_ax(ax, ttl, start, end):
    t = end - start
    ax.axvline(n_frames_pre, lw=2, color='k', ls='--', alpha=.3, zorder=-1)

    ax.set(title=ttl, xticks=[0, n_frames_pre, t], 
                    xlim = [0, t],
                    xticklabels=[-n_sec_pre, 0, n_sec_post],
                    xlabel='time (s)', ylabel='signal')

def process_sig(sig, start, end, n_sec_post, norm=False, filter=True):
    if filter: # median filter
        sig = medfilt(sig, kernel_size=5)
    if norm: # normalize with baseline
        raise NotImplementedError('need to think about wht the best way to norm this stuff is')
        # baseline = np.mean(sig[start: start + n_frames_pre - 1])
        # return sig / baseline
    return sig

def get_next_tag(frame, tags, max_delay = 1000):
    """
        Selects the first tag to happend after a given frame
        as long as not too long a delay happend
    """
    nxt = tags.loc[(tags.session_frame - frame > 0)&(tags.session_frame - frame < max_delay)]
    if not len(nxt): return None
    else: 
        return nxt.session_frame.values[0]

def get_last_tag(frame, tags, max_delay=500):
    nxt = tags.loc[(tags.session_frame - frame < 0)&(np.abs(tags.session_frame - frame) < max_delay)]
    if not len(nxt): 
        return None
    else: 
        return nxt.session_frame.values[-1]

# ------------------------------ Plotting utils ------------------------------ #
def outline(ax, x, y, color, **kwargs):
    ls = kwargs.pop('ls', '-')
    ax.plot(x, y, color=[.4, .4, .4], lw=6, zorder=-1, ls=ls)
    ax.plot(x, y, color=color, lw=4, ls=ls, **kwargs)


def plot_tracking_trace(ax, tracking, color, label, start, end, mark_frame = None, stim_frame=None, escape_start_frame=None):
    # plot tracking
    ax.plot(tracking.x[start:start+n_frames_pre], tracking.y[start:start+n_frames_pre], color=color, ls=':', lw=4)
    outline(ax, tracking.x[start+n_frames_pre:end], tracking.y[start+n_frames_pre:end], color)

    # mark start and end
    ax.scatter(tracking.x[start+n_frames_pre], tracking.y[start+n_frames_pre], s=150,
            color=color, lw=1, edgecolors=[.2, .2, .2], zorder=99)

    if mark_frame is not None:
        ax.scatter(tracking.x[mark_frame], tracking.y[mark_frame], s=150, marker='^',
                color=color, lw=2, edgecolors='k', zorder=99)

    if stim_frame is not None:
        if stim_frame-start > 0: # don't show if it's too in the past
            ax.scatter(tracking.x[stim_frame], tracking.y[stim_frame], s=150,
                    color='k', lw=2, edgecolors=color, zorder=99)

    if escape_start_frame is not None:
            ax.scatter(tracking.x[stim_frame], tracking.y[stim_frame], s=150, marker='D',
                    color=color, lw=2, edgecolors='k', zorder=99)


def plot_speed_trace(ax, speed, xpos, color, label, start, end, mark_frame = None, stim_frame=None, escape_start_frame=None):
    # plot tracking
    ax.plot(speed[start:start+n_frames_pre], xpos[start:start+n_frames_pre], color=color, ls=':', lw=4)
    outline(ax, speed[start+n_frames_pre:end], xpos[start+n_frames_pre:end], color)

    # mark start and end
    ax.scatter(speed[start+n_frames_pre], xpos[start+n_frames_pre], s=150,
            color=color, lw=1, edgecolors=[.2, .2, .2], zorder=99)

    if mark_frame is not None:
        ax.scatter(speed[mark_frame], xpos[mark_frame], s=150, marker='^',
                color=color, lw=2, edgecolors='k', zorder=99)

    if stim_frame is not None:
        if stim_frame-start > 0: # don't show if it's too in the past
            ax.scatter(speed[stim_frame], xpos[stim_frame], s=150,
                    color='k', lw=2, edgecolors=color, zorder=99)

    if escape_start_frame is not None:
            ax.scatter(speed[stim_frame], xpos[stim_frame], s=150, marker='D',
                    color=color, lw=2, edgecolors='k', zorder=99)

def plot_signal_trace(ax, sig, color, start, end, mark_frame=None, stim_frame=None, escape_start_frame=None):
    t = end-start
    # plot tracking
    ax.plot(sig[start:end], label=str(tag.stim_frame),
        color=[.4, .4, .4], lw=6, zorder=-1)
    ax.plot(sig[start:end], label=str(tag.stim_frame),
        color=color, lw=4)

    if mark_frame is not None:
        ax.scatter(mark_frame-start, sig[mark_frame], s=150,
                color='white', lw=2, edgecolors=color, zorder=99)

    if stim_frame is not None:
        if stim_frame-start>0: # don't show it it's too in the past
            ax.scatter(stim_frame-start, sig[stim_frame], s=150,
                    color='k', lw=2, edgecolors=color, zorder=99)

    if escape_start_frame is not None:
            ax.scatter(escape_start_frame-start, sig[escape_start_frame], s=150, marker='D',
                    color=color, lw=2, edgecolors='k', zorder=99)

# %%

include_sessions = [  # including only sessions with lots of data
    'BF166p3-19JUN19',
    'BF164p2-19JUN24',
    'BF164p2-19JUL15',
    'BF161p1-19JUN04',
    'BF164p1-19JUN05',
    'BF164p2-19JUN26'
]

# TODO look into how to normalize the data

# Params
fps = 30
n_sec_pre = 2
n_sec_post = 3
n_frames_pre = n_sec_pre * fps
n_frames_post = n_sec_post * fps

tag_type = 'VideoTag_B'
event_type = ['Loom_Escape', 'US_Escape', 'LoomUS_Escape']

NORMALIZE = False # if true divide sig by the mean of the sig in n_sec_pre
FILTER = True # if true data are median filtered to remove artefact

event_types = list(manual_tags.event_type.unique())


for mouse, sess, sessname in mouse_sessions:
    # if sessname not in include_sessions: continue
    print(f'Processing {sessname}\n')

    # Get data
    tracking, ang_vel, speed, shelter_distance, signals, nrois, is_rec = get_mouse_session_data(mouse, sess, sessions)
    tags = get_tags_by(mouse=mouse, sess_name=sess, event_type=event_type, tag_type=tag_type)
    # print(get_tags_by(mouse=mouse, sess_name=sess, tag_type=tag_type).event_type.unique())
    # continue

    if not len(tags):
        print(f'Didnt find any tags of tupe {tag_type} in session {sessname}')
        continue

    at_shelt_tags = get_tags_by(mouse=mouse, sess_name=sess, event_type=event_type, tag_type='VideoTag_E')
    escape_start_tags = get_tags_by(mouse=mouse, sess_name=sess, event_type=event_type, tag_type='VideoTag_L')
    failures_tags = get_tags_by(mouse=mouse, sess_name=sess, event_type=event_type, tag_type='VideoTag_D')
    
    # Prep figure
    rows, cols = calc_nrows_ncols(len(signals) + 2, (30, 15))
    f, axarr = plt.subplots(ncols=cols, nrows=rows, figsize = (30, 15))
    f.suptitle(f'{sessname} traces aligned to {tag_type}')
    axarr = axarr.flatten()

    # make colors
    colors = []
    for n, (i, tag) in enumerate(tags.iterrows()):
        if 'Loom' in tag.event_type:
            cmap = CMAP
        elif 'US' in tag.event_type:
            cmap = CMAP2
        else:
            raise NotImplementedError
        colors.append(colorMap(n, name=cmap.mpl_colormap, vmin=-1, vmax=len(tags)+1))

    # Plot data
    for count, (color, (i, tag)) in enumerate(zip(colors, tags.iterrows())):
        start = tag.session_frame - n_frames_pre
        end = tag.session_frame + n_frames_post

        if np.sum(is_rec[start:end]) == 0: # recording was off
            continue

        nxt_at_shelt = get_next_tag(tag.session_frame, at_shelt_tags)
        escape_start = get_last_tag(tag.session_frame, escape_start_tags)
        failed = get_next_tag(tag.session_frame, failures_tags)
        if failed:
            print('Failure alarm!!')

        if start > len(tracking):
            print(f'!! tag frame {start} is out of bounds: # frames: {len(tracking)}')
            raise ValueError

        # PLOT TRACKING
        plot_tracking_trace(axarr[0], tracking, color, f'frame {start}', start, end,
                                 mark_frame=nxt_at_shelt, 
                                 stim_frame=tag.session_stim_frame,
                                 escape_start_frame = escape_start,
                                 )
        
        # PLOT SPEED and S distance
        plot_speed_trace(axarr[1], pxperframe_to_cmpersec(speed), tracking.x, 
                                color, f'frame {start}', start, end,
                                 mark_frame=nxt_at_shelt, 
                                 stim_frame=tag.session_stim_frame,
                                 escape_start_frame = escape_start,
                                 )
        # PLOT SIGNALS
        for roin, sig in enumerate(signals):
            if np.std(sig[start:end]) == 0:
                raise ValueError

            plot_signal_trace(axarr[roin+2], 
                                process_sig(sig, start, end, n_sec_pre, norm=NORMALIZE, filter=FILTER), 
                                color, start, end,
                                mark_frame=nxt_at_shelt,
                                stim_frame = tag.session_stim_frame,
                                escape_start_frame = escape_start,
                                )

        # MAKE CUSTOM LEGEND
        if count == 0:
            legend_elements = [
                Line2D([0], [0], color=color, lw=4, marker='D', label='Turn', markersize=10, markeredgecolor='k'),
                Line2D([0], [0], color=color, lw=4, marker='o', label='Run', markersize=10, markeredgecolor='k'),
                Line2D([0], [0], color=color, lw=4, marker='^', label='Shelter', markersize=10, markeredgecolor='k'),
                Line2D([0], [0], color=color, lw=4, label=str(tag.stim_frame)),
            ]
        else:
            legend_elements.append(Line2D([0], [0], color=color, lw=4, label=str(tag.stim_frame)),)

    # fig figure
    for n, ax in enumerate(axarr[2:]):
        if n >= len(signals):
            ax.axis('off')
        else:
            make_signal_ax(ax, f'ROI {n}', start, end)

    axarr[0].set(title='tracking', xlabel='X (px)', ylabel='Y (px)')
    axarr[0].legend(handles=legend_elements)
    axarr[1].set(title='tracking', xlabel='speed (cm/s)', ylabel='X (px) ')

    clean_axes(f)
    set_figure_subplots_aspect(right=.85, top=.9, wspace=.4)
    save_figure(f, str(fld / f'{sessname}_tag_{tag_type[-1]}'), svg=False)

    # break


with open(str(fld / '00_info.txt'), 'w') as out:
    out.write('INFO')
    out.write(f'''IMAGES GENERATED WITH PARAMS:
                        fps = {fps }
                        n_sec_pre = {n_sec_pre}
                        n_sec_post = {n_sec_post}
                        tag_type = {tag_type}
                        event_type = {event_type}
                        NORMALIZE = {NORMALIZE}
                        FILTER = {FILTER}
''')



# %%
