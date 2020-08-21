# %%
import matplotlib.pyplot as plt


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

print(manual_tags.head())

# %%
event_types = list(manual_tags.event_type.unique())

f, ax = plt.subplots()
for mouse, sess, sessname in mouse_sessions:
    tracking, ang_vel, speed, shelter_distance, signals, nrois, is_rec = get_mouse_session_data(mouse, sess, sessions)

    # for ev in event_types:

    tags = get_tags_by(mouse=mouse, sess_name=sess, event_type='Loom_Escape', tag_type='VideoTag_B')
    
    for i, tag in tags.iterrows():
        ax.plot(tracking.x[tag.session_frame:tag.session_frame + 300], tracking.y[tag.session_frame:tag.session_frame + 300],
                label=str(tag.stim_frame))
    break
ax.legend()
# %%
