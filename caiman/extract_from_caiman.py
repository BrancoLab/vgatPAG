import pyinspect as pi
import numpy as np
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.utils.visualization import get_contours
from pathlib import Path
import pickle

from utils import start_server


pi.install_traceback()
print('ready')

fld = Path(r'D:\Dropbox (UCL - SWC)\Project_vgatPAG\analysis\doric\BF166p3_dPAG\19JUN19')
cnm_name = fld/'19JUN19_BF166p3_ds126_crop3_ffcDiv_cnm.hdf5'

if not cnm_name.exists():
    raise FileExistsError(f'Could not find cnm: {str(cnm_name)}')

c, dview, n_processes = start_server()
cnm = load_CNMF(cnm_name, n_processes=n_processes, dview=dview)

# Spatial components: in a d1 x d2 x n_components matrix
A = np.reshape(cnm.estimates.A.toarray(), list(cnm.estimates.dims)+[-1], order='F') # set of spatial footprints

np.save(fld/'A.npy', A)

conts = get_contours(cnm.estimates.A.toarray(), cnm.estimates.dims)

with open(str(fld/'all_contour_data.pkl'), 'wb') as fp:
    pickle.dump(conts, fp)

pi.ok('Extracted A from cnm', f'Folder \n{fld.parent.name}/{fld.name}')
