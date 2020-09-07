from fcutils.video.utils  import get_cap_from_file, get_video_params
 
path = 'Z:\\swc\\branco\\Federico\\ZM_20200904_1098493_video.avi'

print(get_video_params(get_cap_from_file(path)))