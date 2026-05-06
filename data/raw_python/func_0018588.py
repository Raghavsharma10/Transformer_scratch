def packet_get_samples_per_frame(data, fs):
    """Gets the number of samples per frame from an Opus packet"""

    data_pointer = ctypes.c_char_p(data)

    result = _packet_get_nb_frames(data_pointer, ctypes.c_int(fs))
    if result < 0:
        raise OpusError(result)

    return result