import numpy as  np

from behaviour.utilities.signals import get_times_signal_high_and_low



# --------------------- Extract events from tracking data -------------------- #
def get_spont_homings(shelter_distance, speed, astims, vstims, max_duration, min_time_after_stim, initiation_speed_th=4):
    """
        Gets spontaneous runs from the far end to the arena to the shelter. 
        Only keeps runs that meet a bunch of criteria like duration etc.

        :param shelter_distance: np.array with distance from shelter at every frame
        :param speed: np.array with running speed at every frame
        :param astims: list with onset of all audio stimuli
        :param vstims: list with onset of all visual stimuli
        :param max_duration: int, max duration of an homing, in frames
        :param min_time_after_stim:  int, ignore homings that happened within this number of frames from a stim
        :param initiation_speed_th: int, consider the start of an homing when speed first surpasses this value
        """

    # Group all stimuli together
    stims = astims + vstims

    # Get when the mouse enters the ROI (end of arena) or shelter
    in_roi = np.zeros_like(shelter_distance)
    in_roi[shelter_distance > 450] = 1
    ins, outs = get_times_signal_high_and_low(in_roi, th=.5, min_time_between_highs=30)

    in_shelt = np.zeros_like(shelter_distance)
    in_shelt[shelter_distance <=100] = 1
    ins_shelt, outs_shelt = get_times_signal_high_and_low(in_shelt, th=.5, min_time_between_highs=30)

    # Get when the speed is above a given threshold
    speed_th = np.zeros_like(speed)
    speed_th[speed > 4] = 1

    # Loop over all the times the mouse leaves the ROI and keep only runs that meet the criteria
    starts, good_outs = [], []
    for n, roi_out in enumerate(outs):
        # Check if it reaches the shelter 
        next_in_shelt = [i for i in ins_shelt if i>roi_out]
        if not next_in_shelt: continue
        else:
            next_in_shelt = next_in_shelt[0]

        # skip if there will be another out before an in shelter
        if roi_out != outs[-1]:
            if next_in_shelt > outs[n+1]: continue 

        # Skip if the event is too slow
        if next_in_shelt - roi_out > max_duration: 
            continue 

        # Skip if there's a stimulus too close to the event
        stims_range = [s for s in stims if np.abs(s - roi_out) < min_time_after_stim] 
        if stims_range: continue

        # Detect the run onset based on thresholded speed
        fast = np.where(speed_th[:roi_out] == 1)[0][::-1]
        start = fast[0]
        for f in fast:
            if start - f <= 1:
                start = f
            else:
                break

        # Check that there's no errors 
        if shelter_distance[start] < 450: continue

        # Keep the good stuff and return
        starts.append(start)
        good_outs.append(roi_out)
    return starts


def get_spont_out_runs(shelter_distance, speed, astims, vstims, max_duration, min_time_after_stim, initiation_speed_th=4):
    """
        Gets spontaneous runs from the far end to the arena to the shelter. 
        Only keeps runs that meet a bunch of criteria like duration etc.

        :param shelter_distance: np.array with distance from shelter at every frame
        :param speed: np.array with running speed at every frame
        :param astims: list with onset of all audio stimuli
        :param vstims: list with onset of all visual stimuli
        :param max_duration: int, max duration of an homing, in frames
        :param min_time_after_stim:  int, ignore homings that happened within this number of frames from a stim
        :param initiation_speed_th: int, consider the start of an homing when speed first surpasses this value
        """

    # Group all stimuli together
    stims = astims + vstims

    # Get when the mouse enters the ROI (end of arena) or shelter
    in_roi = np.zeros_like(shelter_distance)
    in_roi[shelter_distance > 450] = 1
    ins, outs = get_times_signal_high_and_low(in_roi, th=.5, min_time_between_highs=30)

    in_shelt = np.zeros_like(shelter_distance)
    in_shelt[shelter_distance <=100] = 1
    ins_shelt, outs_shelt = get_times_signal_high_and_low(in_shelt, th=.5, min_time_between_highs=30)

    # Get when the speed is above a given threshold
    speed_th = np.zeros_like(speed)
    speed_th[speed > 4] = 1

    # Loop over all the times the mouse leaves the ROI and keep only runs that meet the criteria
    starts = []
    for shelt_out in outs_shelt:
        # Check if it reaches the shelter 
        next_in_roi = [i for i in ins if i>shelt_out]
        if not next_in_roi: continue
        else:
            next_in_roi = next_in_roi[0]


        # Skip if theevent is too slow
        if next_in_roi - shelt_out > max_duration: 
            continue 

        # Skip if there's a stimulus too close to the event
        stims_range = [s for s in stims if np.abs(s - shelt_out) < min_time_after_stim] 
        if stims_range: continue

        # Detect the run onset based on thresholded speed
        fast = np.where(speed_th[:shelt_out] == 1)[0][::-1]
        start = fast[0]
        for f in fast:
            if start - f <= 1:
                start = f
            else:
                break

        # Check that there's no errors 
        if shelter_distance[start] > 100: continue

        # Keep the good stuff and return
        starts.append(start)
    return starts


