"""
    Paths to folders and files, used throughout the analysis but user specific
"""

import os
import sys

from fcutils.file_io.utils import get_subdirs

if sys.platform == "darwin":
    metadatafile = "vgatPAG/database/metadata.yml"
else:
    metadatafile = "vgatPAG\database\metadata.yml"


# ---------------------------------------------------------------------------- #
#                                RAW DATA FOLDERS                              #
# ---------------------------------------------------------------------------- #
# Main dropbox folder with doric calcium data
if sys.platform != "darwin":
    main_data_folder = 'D:\\Dropbox (UCL - SWC)\\Project_vgatPAG\\analysis\\doric'
    main_code_folder = "D:\\Dropbox (UCL - SWC)\\Project_vgatPAG\\code"
else:
    main_data_folder = '/Users/federicoclaudi/Dropbox (UCL - SWC)/Project_vgatPAG/analysis/doric'
    main_code_folder = '/Users/federicoclaudi/Dropbox (UCL - SWC)/Project_vgatPAG/code'


# Get the path to each mouse's folder
mice_folders = [d for d in get_subdirs(main_data_folder) if 'BF1' in d] 

# ---------------------------------------------------------------------------- #
#                                  DEEPLABCUT                                  #
# ---------------------------------------------------------------------------- #
dlc_config_file = os.path.join(main_code_folder, 'DLC', 'vgatPAG-federico-2020-02-25', 'config.yaml')


# ---------------------------------------------------------------------------- #
#                                 SUMMARY DATA                                 #
# ---------------------------------------------------------------------------- #
summary_file = os.path.join(main_data_folder, 'VGAT_summary', 'VGAT_summary.hdf5')


# ---------------------------------------------------------------------------- #
#                                    OUTPUT                                    #
# ---------------------------------------------------------------------------- #
output_fld = os.path.join(main_data_folder, 'Fede', 'plots')