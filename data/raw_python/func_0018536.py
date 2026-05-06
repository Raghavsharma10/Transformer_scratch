def encode(encoder, pcm, frame_size, max_data_bytes):
    """Encodes an Opus frame

    Returns string output payload
    """

    pcm = ctypes.cast(pcm, c_int16_pointer)
    data = (ctypes.c_char * max_data_bytes)()

    result = _encode(encoder, pcm, frame_size, data, max_data_bytes)
    if result < 0:
        raise OpusError(result)

    return array.array('c', data[:result]).tostring()