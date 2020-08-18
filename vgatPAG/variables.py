from fcutils.plotting.colors import *


# ---------------------------------- GENERAL --------------------------------- #
miniscope_fps = 10
shelter_width_px = 450

# ------------------------------- FOR PLOTTING ------------------------------- #
tc_colors = {
    'US_Escape':darkseagreen,
    'US_noEscape':lawngreen,

    'Loom_Escape':deepskyblue,
    'Loom_noEscape':lightskyblue,

    'None_Run':darkmagenta,
    'None_Escape':lilla,
    }

stims_colors = dict(
    visual=seagreen,
    audio=salmon
)

spont_events_colors = dict(
    homing=magenta,
    homing_peak_speed =magenta,
    outrun=orange
)