def read_spec(filename, fname='', **kwargs):
    """Read FITS or ASCII spectrum.

    Parameters
    ----------
    filename : str or file pointer
        Spectrum file name or pointer.

    fname : str
        Filename. This is *only* used if ``filename`` is a pointer.

    kwargs : dict
        Keywords acceptable by :func:`read_fits_spec` (if FITS) or
        :func:`read_ascii_spec` (if ASCII).

    Returns
    -------
    header : dict
        Metadata.

    wavelengths, fluxes : `~astropy.units.quantity.Quantity`
        Wavelength and flux of the spectrum.

    Raises
    ------
    synphot.exceptions.SynphotError
        Read failed.

    """
    if isinstance(filename, str):
        fname = filename
    elif not fname:  # pragma: no cover
        raise exceptions.SynphotError('Cannot determine filename.')

    if fname.endswith('fits') or fname.endswith('fit'):
        read_func = read_fits_spec
    else:
        read_func = read_ascii_spec

    return read_func(filename, **kwargs)