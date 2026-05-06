def packet_get_nb_frames(data, length=None):
    """Gets the number of frames in an Opus packet"""

    data_pointer = ctypes.c_char_p(data)
    if length is None:
        length = len(data)

    result = _packet_get_nb_frames(data_pointer, ctypes.c_int(length))
    if result < 0:
        raise OpusError(result)

    return result