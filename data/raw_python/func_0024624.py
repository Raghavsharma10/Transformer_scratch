def read_fits_spec(filename, ext=1, wave_col='WAVELENGTH', flux_col='FLUX',
                   wave_unit=u.AA, flux_unit=units.FLAM):
    """Read FITS spectrum.

    Wavelength and flux units are extracted from ``TUNIT1`` and ``TUNIT2``
    keywords, respectively, from data table (not primary) header.
    If these keywords are not present, units are taken from
    ``wave_unit`` and ``flux_unit`` instead.

    Parameters
    ----------
    filename : str or file pointer
        Spectrum file name or pointer.

    ext: int
        FITS extension with table data. Default is 1.

    wave_col, flux_col : str
        Wavelength and flux column names (case-insensitive).

    wave_unit, flux_unit : str or `~astropy.units.core.Unit`
        Wavelength and flux units, which default to Angstrom and FLAM,
        respectively. These are *only* used if ``TUNIT1`` and ``TUNIT2``
        keywords are not present in table (not primary) header.

    Returns
    -------
    header : dict
        Primary header only. Extension header is discarded.

    wavelengths, fluxes : `~astropy.units.quantity.Quantity`
        Wavelength and flux of the spectrum.

    """
    fs = fits.open(filename)
    header = dict(fs[str('PRIMARY')].header)
    wave_dat = fs[ext].data.field(wave_col).copy()
    flux_dat = fs[ext].data.field(flux_col).copy()
    fits_wave_unit = fs[ext].header.get('TUNIT1')
    fits_flux_unit = fs[ext].header.get('TUNIT2')

    if fits_wave_unit is not None:
        try:
            wave_unit = units.validate_unit(fits_wave_unit)
        except (exceptions.SynphotError, ValueError) as e:  # pragma: no cover
            warnings.warn(
                '{0} from FITS header is not valid wavelength unit, using '
                '{1}: {2}'.format(fits_wave_unit, wave_unit, e),
                AstropyUserWarning)

    if fits_flux_unit is not None:
        try:
            flux_unit = units.validate_unit(fits_flux_unit)
        except (exceptions.SynphotError, ValueError) as e:  # pragma: no cover
            warnings.warn(
                '{0} from FITS header is not valid flux unit, using '
                '{1}: {2}'.format(fits_flux_unit, flux_unit, e),
                AstropyUserWarning)

    wave_unit = units.validate_unit(wave_unit)
    flux_unit = units.validate_unit(flux_unit)

    wavelengths = wave_dat * wave_unit
    fluxes = flux_dat * flux_unit

    if isinstance(filename, str):
        fs.close()

    return header, wavelengths, fluxes