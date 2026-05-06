def src_new(converter_type, channels):
    """Initialise a new sample rate converter.

    Parameters
    ----------
    converter_type : int
        Converter to be used.
    channels : int
        Number of channels.

    Returns
    -------
    state
        An anonymous pointer to the internal state of the converter.
    error : int
        Error code.
    """
    error = ffi.new('int*')
    state = _lib.src_new(converter_type, channels, error)
    return state, error[0]