def decode(decoder, data, length, frame_size, decode_fec, channels=2):
    """Decode an Opus frame

    Unlike the `opus_decode` function , this function takes an additional parameter `channels`,
    which indicates the number of channels in the frame
    """

    pcm_size = frame_size * channels * ctypes.sizeof(ctypes.c_int16)
    pcm = (ctypes.c_int16 * pcm_size)()
    pcm_pointer = ctypes.cast(pcm, c_int16_pointer)

    # Converting from a boolean to int
    decode_fec = int(bool(decode_fec))

    result = _decode(decoder, data, length, pcm_pointer, frame_size, decode_fec)
    if result < 0:
        raise OpusError(result)

    return array.array('h', pcm).tostring()