def src_callback_read(state, ratio, frames, data):
    """Read up to `frames` worth of data using the callback API.

    Returns
    -------
    frames : int
        Number of frames read or -1 on error.
    """
    data_ptr = ffi.cast('float*f', ffi.from_buffer(data))
    return _lib.src_callback_read(state, ratio, frames, data_ptr)