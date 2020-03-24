# %%

from fcutils.file_io.io import open_hdf

# %%
summary_file = 'D:\\Dropbox (UCL - SWC)\\Project_vgatPAG\\analysis\\doric\\VGAT_summary\\VGAT_summary.hdf5'


f, keys, subkeys = open_hdf(summary_file)

# %%
exp = dict(f['all']['LoomUS_Escape'])
keys = list(exp.keys())

# %%
trial_class = 'LoomUS_Escape'

# %%


# %%
trials[sessions[0]]['frame_115070']
# %%
