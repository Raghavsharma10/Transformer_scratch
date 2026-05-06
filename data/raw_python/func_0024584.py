def validate_wavelengths(wavelengths):
    """Check wavelengths for ``synphot`` compatibility.

    Wavelengths must satisfy these conditions:

        * valid unit type, if given
        * no zeroes
        * monotonic ascending or descending
        * no duplicate values

    Parameters
    ----------
    wavelengths : array-like or `~astropy.units.quantity.Quantity`
        Wavelength values.

    Raises
    ------
    synphot.exceptions.SynphotError
        Wavelengths unit type is invalid.

    synphot.exceptions.DuplicateWavelength
        Wavelength array contains duplicate entries.

    synphot.exceptions.UnsortedWavelength
        Wavelength array is not monotonic.

    synphot.exceptions.ZeroWavelength
        Negative or zero wavelength occurs in wavelength array.

    """
    if isinstance(wavelengths, u.Quantity):
        units.validate_wave_unit(wavelengths.unit)
        wave = wavelengths.value
    else:
        wave = wavelengths

    if np.isscalar(wave):
        wave = [wave]

    wave = np.asarray(wave)

    # Check for zeroes
    if np.any(wave <= 0):
        raise exceptions.ZeroWavelength(
            'Negative or zero wavelength occurs in wavelength array',
            rows=np.where(wave <= 0)[0])

    # Check for monotonicity
    sorted_wave = np.sort(wave)
    if not np.alltrue(sorted_wave == wave):
        if np.alltrue(sorted_wave[::-1] == wave):
            pass  # Monotonic descending is allowed
        else:
            raise exceptions.UnsortedWavelength(
                'Wavelength array is not monotonic',
                rows=np.where(sorted_wave != wave)[0])

    # Check for duplicate values
    if wave.size > 1:
        dw = sorted_wave[1:] - sorted_wave[:-1]
        if np.any(dw == 0):
            raise exceptions.DuplicateWavelength(
                'Wavelength array contains duplicate entries',
                rows=np.where(dw == 0)[0])