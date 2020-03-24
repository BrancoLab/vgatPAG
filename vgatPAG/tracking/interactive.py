# %%

import os
import deeplabcut as dlc



# %%
dlc_config_file = "D:\\Dropbox (UCL - SWC)\\Project_vgatPAG\\code\\DLC\\vgatPAG-federico-2020-02-25\\config.yaml"
dlc_fld = "D:\\Dropbox (UCL - SWC)\\Project_vgatPAG\\code\\DLC"


# %%
clips = [os.path.join(dlc_fld, v) for v in os.listdir(dlc_fld) if 'clip' in v and 'cam' not in v]
dlc.add_new_videos(dlc_config_file, clips, copy_videos=True)