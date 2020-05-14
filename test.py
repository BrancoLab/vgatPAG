from vgatPAG.database.db_tables import TiffTimes
import matplotlib.pyplot as plt
import numpy as np

# TiffTimes.drop()
TiffTimes.populate(display_progress=True)



# import matplotlib.pyplot as plt

# f, ax = plt.subplots()

# ax.plot(microscope_triggers)

# for start, stop in startends:
#     ax.axvline(start, color='red')
#     ax.axvline(stop, color='green')
# ax.set(xlim=[0, 27028012])
# plt.show()

        # starts, ends = get_times_signal_high_and_low(microscope_triggers)
        # starts = (starts / self.sampling_frequency * fps) # .astype(np.int64)
        # ends = (ends / self.sampling_frequency * fps) # .astype(np.int64)

        # cam_starts, cam_ends = get_times_signal_high_and_low(camera_triggers)
        # cam_starts = (cam_starts / self.sampling_frequency * fps).astype(np.int64)
        # cam_ends = (cam_ends / self.sampling_frequency * fps).astype(np.int64)
        
        # # Get an example ROI signals
        # roi_data = f['CRaw_ROI_1'][()]
        # n_frames = len(roi_data)
        
        
        # signal = np.zeros_like(roi_data)
        # status = 'low'5
        # idx = 0
        # startends = [] # will store tuples with start and end of each recording
        # pair = []
        # itern = 0
        # while idx < n_frames:
        #     itern += 1
        #     if itern > n_frames: break

        #     if status == 'low':
        #         # Go to next frame start
        #         idx = starts[starts > idx][0]
                
        #         # Keep track of stuff
        #         pair.append(idx)
        #         status = 'high'

                

        #     elif status == 'high':
        #         # Go to next end frame
        #         try:
        #             idx = ends[ends > idx][0]
        #         except :
        #             # We've reached the end
        #             break


        #         # look at the time after the frame to see if there's 30s of silence
        #         sample_idx = np.int32(((idx+1) / fps) * self.sampling_frequency)
        #         after = microscope_triggers[sample_idx: sample_idx + 30 * self.sampling_frequency]

        #         if np.std(after)  <= .1:
        #             pair.append(idx)
        #             startends.append(tuple(pair.copy()))
        #             print('Added pair: ', pair)
        #             pair = []
        #             status = 'low'
                    
        # a = 1
