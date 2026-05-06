def _check_data(data):
    """Check whether `data` is a valid input/output for libsamplerate.

    Returns
    -------
    num_frames
        Number of frames in `data`.
    channels
        Number of channels in `data`.

    Raises
    ------
        ValueError: If invalid data is supplied.
    """
    if not (data.dtype == _np.float32 and data.flags.c_contiguous):
        raise ValueError('supplied data must be float32 and C contiguous')
    if data.ndim == 2:
        num_frames, channels = data.shape
    elif data.ndim == 1:
        num_frames, channels = data.size, 1
    else:
        raise ValueError('rank > 2 not supported')
    return num_frames, channels