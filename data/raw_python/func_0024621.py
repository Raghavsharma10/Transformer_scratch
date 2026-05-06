def read_remote_spec(filename, encoding='binary', cache=True,
                     show_progress=True, **kwargs):
    """Read FITS or ASCII spectrum from a remote location.

    Parameters
    ----------
    filename : str
        Spectrum filename.

    encoding, cache, show_progress
        See :func:`~astropy.utils.data.get_readable_fileobj`.

    kwargs : dict
        Keywords acceptable by :func:`read_fits_spec` (if FITS) or
        :func:`read_ascii_spec` (if ASCII).

    Returns
    -------
    header : dict
        Metadata.

    wavelengths, fluxes : `~astropy.units.quantity.Quantity`
        Wavelength and flux of the spectrum.

    """
    with get_readable_fileobj(filename, encoding=encoding, cache=cache,
                              show_progress=show_progress) as fd:
        header, wavelengths, fluxes = read_spec(fd, fname=filename, **kwargs)

    return header, wavelengths, fluxes