def src_callback_new(callback, converter_type, channels):
    """Initialisation for the callback based API.

    Parameters
    ----------
    callback : function
        Called whenever new frames are to be read. Must return a NumPy array
        of shape (num_frames, channels).
    converter_type : int
        Converter to be used.
    channels : int
        Number of channels.

    Returns
    -------
    state
        An anonymous pointer to the internal state of the converter.
    handle
        A CFFI handle to the callback data.
    error : int
        Error code.

    """
    cb_data = {'callback': callback, 'channels': channels}
    handle = ffi.new_handle(cb_data)
    error = ffi.new('int*')
    state = _lib.src_callback_new(_src_input_callback, converter_type,
                                  channels, error, handle)
    if state == ffi.NULL:
        return None, handle, error[0]
    return state, handle, error[0]