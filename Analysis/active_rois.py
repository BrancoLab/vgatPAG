import pandas as pd
import numpy as np

curated = [
    ('BF118p2_18NOV29__Fiji_ROI_12_active_True', '+'),
    ('BF118p2_18NOV29__Fiji_ROI_14_active_True', '-'),
    ('BF118p2_18NOV29__Fiji_ROI_16_active_True', '+'),
    ('BF118p2_18NOV29__Fiji_ROI_17_active_True', '+'),
    ('BF118p2_18NOV29__Fiji_ROI_21_active_True', '+'),
    ('BF118p2_18NOV29__Fiji_ROI_22_active_True', '+'),
    ('BF118p2_18NOV29__Fiji_ROI_25_active_True', '-'),
    ('BF118p2_18SEP18__Fiji_ROI_7_active_True', '-'),
    ('BF118p2_18SEP18__Fiji_ROI_22_active_True', '+'),
    ('BF118p2_18SEP18__Fiji_ROI_23_active_True', '-'),
    ('BF118p2_18SEP18__Fiji_ROI_25_active_True', '-'),
    ('BF164p1_19JUN05__Fiji_ROI_2_active_True', '+'),
    ('BF118p2_18SEP27__Fiji_ROI_12_active_True', '+'),
    ('BF118p2_18SEP27__Fiji_ROI_26_active_True', '-'),
    ('BF121p2_18SEP28__Fiji_ROI_11_active_True', '+'),
    ('BF136p1_19FEB04__Fiji_ROI_16_active_True', '+'),
    ('BF136p1_19FEB04__Fiji_ROI_19_active_True', '-'),
    ('BF136p1_19FEB04__Fiji_ROI_24_active_True', '+'),
    ('BF136p1_19FEB04__Fiji_ROI_27_active_True', '+'),
    ('BF166p3_19JUN05__Fiji_ROI_12_active_True', '+'),
    ('BF166p3_19JUN05__Fiji_ROI_16_active_True', '-'),
    ('BF166p3_19JUN05__Fiji_ROI_24_active_True', '-'),
    ('BF166p3_19JUN05__Fiji_ROI_25_active_True', '-'),
    ('BF166p3_19JUN19__Fiji_ROI_2_active_True', '-'),
    ('BF166p3_19JUN19__Fiji_ROI_6_active_True', '-'),
    ('BF166p3_19JUN19__Fiji_ROI_12_active_True', '-'),
    ('BF118p2_18SEP27__Fiji_ROI_21_active_True', '-'),
    ('BF118p2_18SEP27__Fiji_ROI_22_active_True', '-'),
    ('BF118p2_18SEP27__Fiji_ROI_24_active_True', '-'),
    ('BF118p2_18SEP27__Fiji_ROI_28_active_True', '-'),
    ('BF121p2_18SEP28__Fiji_ROI_22_active_True', '-'),
    ('BF136p1_19FEB25__Fiji_ROI_5_active_True', '-'),
    ('BF121p2_18OCT19__Fiji_ROI_6_active_True', '-'),
    ('BF121p2_18OCT19__Fiji_ROI_25_active_True', '-'),
    ('BF121p2_18SEP28__Fiji_ROI_10_active_True', '-'),
    ('BF136p1_19FEB25__Fiji_ROI_10_active_True', '-'),
    ('BF161p1_19JUN03__Fiji_ROI_1_active_True', '-'),
    ('BF161p1_19JUN03__Fiji_ROI_13_active_True', '-'),
    ('BF164p2_19JUL15__Fiji_ROI_7_active_True', '-'),
    ('BF164p2_19JUN26__Fiji_ROI_13_active_True', '-'),
    ('BF166p3_19JUN05__Fiji_ROI_5_active_True', '-'),
]

def manual_select_active():
    # clean up manually curated list
    cur = dict(date=[], mouse=[], roi=[], keep=[])
    for roi, keep in curated:
        mouse, date, = roi.split('_')[:2]
        rid = roi.split('__')[1].split('_active')[0]
        cur['date'].append(date)
        cur['mouse'].append(mouse)
        cur['roi'].append(rid)
        cur['keep'].append(True if keep=='+' else False)
    cur = pd.DataFrame(cur)

    # curate active ROIs
    selected = dict(date=[], mouse=[], roi=[], active=[])
    active = pd.read_hdf('ACTIVE_ROIS.h5', key='hdf')

    A = []
    for i, roi in active.iterrows():
        roi_cur = cur.loc[(cur.mouse==roi.mouse)&(cur.date==roi.date)&(cur.roi==roi.roi)]

        if roi_cur.empty:
            keep = roi.active
        else:
            keep = roi_cur.keep.values[0]
        
        selected['date'].append(roi.date)
        selected['mouse'].append(roi.mouse)
        selected['roi'].append(roi.roi)
        selected['active'].append(keep)

        if keep:
            A.append(f'{roi.date} {roi.mouse} {roi.roi}\n')

    with open('active_rois.txt', 'w+') as w:
        w.writelines(A)

    pd.DataFrame(selected).to_hdf('ACTIVE_ROIS_CURATED.h5', key='hdf')

def get_active_rois(rois, sequences, sess):
    active = pd.read_hdf('ACTIVE_ROIS_CURATED.h5', key='hdf')
    active = active.loc[(active.mouse==sess['mouse'])&(active.date==sess['date'])]
    
    active_rois = rois[[r for r in rois.columns if np.all(active.loc[active.roi == r].active)]]
    return active_rois

if __name__ == '__main__':
    manual_select_active()