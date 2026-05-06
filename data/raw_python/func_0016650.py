def _src_input_callback(cb_data, data):
    """Internal callback function to be used with the callback API.

    Pulls the Python callback function from the handle contained in `cb_data`
    and calls it to fetch frames. Frames are converted to the format required by
    the API (float, interleaved channels). A reference to these data is kept
    internally.

    Returns
    -------
    frames : int
        The number of frames supplied.
    """
    cb_data = ffi.from_handle(cb_data)
    ret = cb_data['callback']()
    if ret is None:
        cb_data['last_input'] = None
        return 0  # No frames supplied
    input_data = _np.require(ret, requirements='C', dtype=_np.float32)
    input_frames, channels = _check_data(input_data)

    # Check whether the correct number of channels is supplied by user.
    if cb_data['channels'] != channels:
        raise ValueError('Invalid number of channels in callback.')

    # Store a reference of the input data to ensure it is still alive when
    # accessed by libsamplerate.
    cb_data['last_input'] = input_data

    data[0] = ffi.cast('float*', ffi.from_buffer(input_data))
    return input_frames