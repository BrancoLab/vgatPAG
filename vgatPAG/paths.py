"""
    Paths to folders and files, used throughout the analysis but user specific
"""

import os

from fcutils.file_io.utils import get_subdirs

metadatafile = "vgatPAG\database\metadata.yml"

# ---------------------------------------------------------------------------- #
#                                RAW DATA FOLDERS                              #
# ---------------------------------------------------------------------------- #
# Main dropbox folder with doric calcium data
main_data_folder = 'D:\\Dropbox (UCL - SWC)\\Project_vgatPAG\\analysis\\doric'

# Get the path to each mouse's folder
mice_folders = [d for d in get_subdirs(main_data_folder) if 'BF1' in d] 

# ---------------------------------------------------------------------------- #
#                                  DEEPLABCUT                                  #
# ---------------------------------------------------------------------------- #
dlc_config_file = "D:\\Dropbox (UCL - SWC)\\Project_vgatPAG\\code\\DLC\\vgatPAG-federico-2020-02-25\\config.yaml"


# ---------------------------------------------------------------------------- #
#                                 SUMMARY DATA                                 #
# ---------------------------------------------------------------------------- #
summary_file = 'D:\\Dropbox (UCL - SWC)\\Project_vgatPAG\\analysis\\doric\\VGAT_summary\\VGAT_summary.hdf5'


# ---------------------------------------------------------------------------- #
#                                    OUTPUT                                    #
# ---------------------------------------------------------------------------- #
output_fld = "D:\\Dropbox (UCL - SWC)\\Project_vgatPAG\\analysis\\doric\\For_Fede\\plots"