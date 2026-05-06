def generate_wavelengths(minwave=500, maxwave=26000, num=10000, delta=None,
                         log=True, wave_unit=u.AA):
    """Generate wavelength array to be used for spectrum sampling.

    .. math::

        minwave \\le \\lambda < maxwave

    Parameters
    ----------
    minwave, maxwave : float
        Lower and upper limits of the wavelengths.
        These must be values in linear space regardless of ``log``.

    num : int
        The number of wavelength values.
        This is only used when ``delta=None``.

    delta : float or `None`
        Delta between wavelength values.
        When ``log=True``, this is the spacing in log space.

    log : bool
        If `True`, the wavelength values are evenly spaced in log scale.
        Otherwise, spacing is linear.

    wave_unit : str or `~astropy.units.core.Unit`
        Wavelength unit. Default is Angstrom.

    Returns
    -------
    waveset : `~astropy.units.quantity.Quantity`
        Generated wavelength set.

    waveset_str : str
        Info string associated with the result.

    """
    wave_unit = units.validate_unit(wave_unit)

    if delta is not None:
        num = None

    waveset_str = 'Min: {0}, Max: {1}, Num: {2}, Delta: {3}, Log: {4}'.format(
        minwave, maxwave, num, delta, log)

    # Log space
    if log:
        logmin = np.log10(minwave)
        logmax = np.log10(maxwave)

        if delta is None:
            waveset = np.logspace(logmin, logmax, num, endpoint=False)
        else:
            waveset = 10 ** np.arange(logmin, logmax, delta)

    # Linear space
    else:
        if delta is None:
            waveset = np.linspace(minwave, maxwave, num, endpoint=False)
        else:
            waveset = np.arange(minwave, maxwave, delta)

    return waveset.astype(np.float64) * wave_unit, waveset_str