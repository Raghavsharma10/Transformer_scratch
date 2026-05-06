def packet_get_nb_channels(data):
    """Gets the number of channels from an Opus packet"""

    data_pointer = ctypes.c_char_p(data)

    result = _packet_get_nb_channels(data_pointer)
    if result < 0:
        raise OpusError(result)

    return result