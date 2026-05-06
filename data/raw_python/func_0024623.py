def read_ascii_spec(filename, wave_unit=u.AA, flux_unit=units.FLAM, **kwargs):
    """Read ASCII spectrum.

    ASCII table must have following columns:

        #. Wavelength data
        #. Flux data

    It can have more than 2 columns but the rest is ignored.
    Comments are discarded.

    Parameters
    ----------
    filename : str or file pointer
        Spectrum file name or pointer.

    wave_unit, flux_unit : str or `~astropy.units.core.Unit`
        Wavelength and flux units, which default to Angstrom and FLAM,
        respectively.

    kwargs : dict
        Keywords accepted by :func:`astropy.io.ascii.ui.read`.

    Returns
    -------
    header : dict
        This is just an empty dictionary, so returned values
        are the same as :func:`read_fits_spec`.

    wavelengths, fluxes : `~astropy.units.quantity.Quantity`
        Wavelength and flux of the spectrum.
        They are set to 'float64' percision.

    """
    header = {}

    dat = ascii.read(filename, **kwargs)

    wave_unit = units.validate_unit(wave_unit)
    flux_unit = units.validate_unit(flux_unit)

    wavelengths = dat.columns[0].data.astype(np.float64) * wave_unit
    fluxes = dat.columns[1].data.astype(np.float64) * flux_unit

    return header, wavelengths, fluxes