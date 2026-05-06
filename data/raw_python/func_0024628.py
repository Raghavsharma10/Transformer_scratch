def convert_flux(wavelengths, fluxes, out_flux_unit, **kwargs):
    """Perform conversion for :ref:`supported flux units <synphot-flux-units>`.

    Parameters
    ----------
    wavelengths : array-like or `~astropy.units.quantity.Quantity`
        Wavelength values. If not a Quantity, assumed to be in
        Angstrom.

    fluxes : array-like or `~astropy.units.quantity.Quantity`
        Flux values. If not a Quantity, assumed to be in PHOTLAM.

    out_flux_unit : str or `~astropy.units.core.Unit`
        Output flux unit.

    area : float or `~astropy.units.quantity.Quantity`
        Area that fluxes cover. If not a Quantity, assumed to be in
        :math:`cm^{2}`. This value *must* be provided for conversions involving
        OBMAG and count, otherwise it is not needed.

    vegaspec : `~synphot.spectrum.SourceSpectrum`
        Vega spectrum from :func:`~synphot.spectrum.SourceSpectrum.from_vega`.
        This is *only* used for conversions involving VEGAMAG.

    Returns
    -------
    out_flux : `~astropy.units.quantity.Quantity`
        Converted flux values.

    Raises
    ------
    astropy.units.core.UnitsError
        Conversion failed.

    synphot.exceptions.SynphotError
        Area or Vega spectrum is not given when needed.

    """
    if not isinstance(fluxes, u.Quantity):
        fluxes = fluxes * PHOTLAM

    out_flux_unit = validate_unit(out_flux_unit)
    out_flux_unit_name = out_flux_unit.to_string()
    in_flux_unit_name = fluxes.unit.to_string()

    # No conversion necessary
    if in_flux_unit_name == out_flux_unit_name:
        return fluxes

    in_flux_type = fluxes.unit.physical_type
    out_flux_type = out_flux_unit.physical_type

    # Wavelengths must Quantity
    if not isinstance(wavelengths, u.Quantity):
        wavelengths = wavelengths * u.AA

    eqv = u.spectral_density(wavelengths)

    # Use built-in astropy equivalencies
    try:
        out_flux = fluxes.to(out_flux_unit, eqv)

    # Use PHOTLAM as in-between unit
    except u.UnitConversionError:
        # Convert input unit to PHOTLAM
        if fluxes.unit == PHOTLAM:
            flux_photlam = fluxes
        elif in_flux_type != 'unknown':
            flux_photlam = fluxes.to(PHOTLAM, eqv)
        else:
            flux_photlam = _convert_flux(
                wavelengths, fluxes, PHOTLAM, **kwargs)

        # Convert PHOTLAM to output unit
        if out_flux_unit == PHOTLAM:
            out_flux = flux_photlam
        elif out_flux_type != 'unknown':
            out_flux = flux_photlam.to(out_flux_unit, eqv)
        else:
            out_flux = _convert_flux(
                wavelengths, flux_photlam, out_flux_unit, **kwargs)

    return out_flux