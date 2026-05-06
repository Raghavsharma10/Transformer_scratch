def get_frame_list():
    """
    Create the list of frames
    """

    # TODO: use this function in IPS below (less code duplication)

    frame_info_list = []
    frame_list = []
    frame = inspect.currentframe()
    while frame is not None:
        frame_list.append(frame)
        info = inspect.getframeinfo(frame)
        frame_info_list.append(info)
        frame = frame.f_back

    frame_info_list.reverse()
    frame_list.reverse()
    frame_info_str_list = [format_frameinfo(fi) for fi in frame_info_list]

    return frame_list, frame_info_list, frame_info_str_list