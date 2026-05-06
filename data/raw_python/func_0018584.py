def create(fs, channels):
    """Allocates and initializes a decoder state"""

    result_code = ctypes.c_int()

    result = _create(fs, channels, ctypes.byref(result_code))
    if result_code.value is not 0:
        raise OpusError(result_code.value)

    return result