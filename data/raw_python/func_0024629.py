def _convert_flux(wavelengths, fluxes, out_flux_unit, area=None,
                  vegaspec=None):
    """Flux conversion for PHOTLAM <-> X."""
    flux_unit_names = (fluxes.unit.to_string(), out_flux_unit.to_string())

    if PHOTLAM.to_string() not in flux_unit_names:
        raise exceptions.SynphotError(
            'PHOTLAM must be one of the conversion units but get '
            '{0}.'.format(flux_unit_names))

    # VEGAMAG
    if VEGAMAG.to_string() in flux_unit_names:
        from .spectrum import SourceSpectrum

        if not isinstance(vegaspec, SourceSpectrum):
            raise exceptions.SynphotError('Vega spectrum is missing.')

        flux_vega = vegaspec(wavelengths)

        out_flux = fluxes.to(
            out_flux_unit,
            equivalencies=spectral_density_vega(wavelengths, flux_vega))

    # OBMAG or count
    elif (u.count in (fluxes.unit, out_flux_unit) or
          OBMAG.to_string() in flux_unit_names):
        if area is None:
            raise exceptions.SynphotError(
                'Area is compulsory for conversion involving count or OBMAG.')
        elif not isinstance(area, u.Quantity):
            area = area * AREA

        out_flux = fluxes.to(
            out_flux_unit,
            equivalencies=spectral_density_count(wavelengths, area))

    else:
        raise u.UnitsError('{0} and {1} are not convertible'.format(
            fluxes.unit, out_flux_unit))

    return out_flux