import napari
import os
import numpy as np
import caiman as cm

from fcutils.file_io.utils import check_file_exists


def load_mmapped(filepath):
    check_file_exists(filepath, raise_error=True)

    Yr, dims, T = cm.load_memmap(filepath)
    images = Yr.T.reshape((T,) + dims, order='F')
    return images


def look_at_video(filepath, rgb=False):
    # Use this function to look at a single video in napari
    # filepath should be path to memmapped file
    images = load_mmapped(filepath)

    with napari.gui_qt():
        viewer = napari.view_image(images, rgb=rgb)


_cmaps = [
    "gray",
    "twilight", 
    "hsv",
    "turbo"
]


def compare_videos(rgb=False, contrast_limits=[120, 600], **kwargs):
    """
        Look at a number of videos side by side. 
        Depending on the number of videos and their size it might take a while to 
        load up. 

        Pass a list of paths to memmapped or tiff files as kwargs, with the name of each file
        as they keyword and the path as the argument
    """

    images = {k:load_mmapped(fp) if fp.endswith(".mmap") else napari.utils.io.magic_imread(fp) 
                        for k, fp in kwargs.items()}
    if not images: return

    with napari.gui_qt():
        v = napari.Viewer(ndisplay=2)

        for n, (ttl, imgs) in enumerate(images.items()):
            v.add_image(imgs, name=ttl, colormap=_cmaps[n], contrast_limits=contrast_limits)

        v.grid_view(n_column=len(list(images.keys())))



# > example of how to sue compare videos to look at a few videos at the same time
if __name__ == "__main__":
    fld = "D:\\Dropbox (UCL - SWC)\\Project_vgatPAG\\analysis\\doric\\BF164p1\\19JUN05"
    v1 = os.path.join(fld, '19JUN05_BF164p1_v1_ds126_ffcSub.tif')
    v2 = os.path.join(fld, "19JUN05_BF164p1_v1_ds126_ffcSub_rig__d1_126_d2_126_d3_1_order_C_frames_22662_.mmap")
    v3 = os.path.join(fld, "19JUN05_BF164p1_v1_ds126_ffcSub_els__d1_126_d2_126_d3_1_order_C_frames_22662_.mmap")


    compare_videos(raw=v1, rigid_mc=v2, pw_mc=v3)