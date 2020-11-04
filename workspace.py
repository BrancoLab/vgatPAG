from fcutils.file_io.io import open_hdf, load_yaml
import matplotlib.pyplot as plt

f, keys, subkeys, allkeys = open_hdf(r'D:\Dropbox (UCL)\Project_vgatPAG\analysis\doric\VGAT_summary\fede2.hdf5')

rois = [k for k in keys if 'Fiji_ROI' in k]

print(subkeys, allkeys)

for r in rois:
    print(r)

roi_trace = f[rois[0]][()] 
plt.plot(roi_trace)
plt.show()