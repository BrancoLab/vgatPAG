# %%
from pathlib import Path
from fcutils.file_io.io import open_hdf
import matplotlib.pyplot as plt
from rich import print
from rich.pretty import install
install()


# %%

# --------------------------------- Metadata --------------------------------- #

fld = Path('D:\\Dropbox (UCL)\\Project_vgatPAG\\analysis\\doric\\VGAT_summary\\temp_tagged-mp4')

mice_subs = [f for f in fld.glob('*') if f.is_dir()]

mice = [f.name for f in mice_subs]

experiments = {}
for sub in mice_subs:
hdfs = sorted([f for f in sub.glob('*.hdf5') if 'Fiji-tag' in f.name])
exp_names = [f.name.split('_Fiji')[0] for f in hdfs]

exp_data = {}
for exp in exp_names:
    h = [h for h in hdfs if exp in str(h)]
    v = [f for f in sub.glob('*.mp4') if exp in str(f)]

    if len(h)!=1 or len(v)!=1:
        raise ValueError()

    exp_data[exp] = dict(hdf=h[0], video=v[0])
    experiments[sub.name] = exp_data
    break


print(exp_data)


# %%
keys
# %%
len(f['Fiji_ROI_10'][()])
# %%
plt.plot(f['Real-time Coordinates']['X-Vertical'][()], f['Real-time Coordinates']['Y-Horizontal'][()])

# %%
# TODO Get metadata
# TODO Get tracking
# TODO Get raw troi trace and tracking in same time stamps stuff
# TODO DFF

# %%
1291 * 30
# %%
