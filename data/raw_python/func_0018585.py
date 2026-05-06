def packet_get_bandwidth(data):
    """Gets the bandwidth of an Opus packet."""

    data_pointer = ctypes.c_char_p(data)

    result = _packet_get_bandwidth(data_pointer)
    if result < 0:
        raise OpusError(result)

    return result