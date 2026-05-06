def snr(*args, **kwargs):
    """Compute the SNR of binaries.

    snr is a function that takes binary parameters and sensitivity curves as inputs,
    and returns snr for chosen phases.

    Warning: All binary parameters must be either scalar, len-1 arrays,
    or arrays of the same length. All of these can be used at once. However,
    you cannot input multiple arrays of different lengths.

    Arguments:
        *args: Arguments for :meth:`gwsnrcalc.utils.pyphenomd.PhenomDWaveforms.__call__`
        **kwargs: Keyword arguments related to
            parallel generation (see :class:`gwsnrcalc.utils.parallel`),
            waveforms (see :class:`gwsnrcalc.utils.pyphenomd`),
            or sensitivity information (see :class:`gwsnrcalc.utils.sensitivity`).

    Returns:
        (dict or list of dict): Signal-to-Noise Ratio dictionary for requested phases.

    """
    squeeze = False
    max_length = 0
    for arg in args:
        try:
            length = len(arg)
            if length > max_length:
                max_length = length

        except TypeError:
            pass

    if max_length == 0:
        squeeze = True

    kwargs['length'] = max_length

    snr_main = SNR(**kwargs)
    if squeeze:
        snr_out = snr_main(*args)
        return {key: np.squeeze(snr_out[key]) for key in snr_out}
    return snr_main(*args)