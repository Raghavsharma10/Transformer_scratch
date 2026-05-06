def spectral_density_vega(wav, vegaflux):
    """Flux equivalencies between PHOTLAM and VEGAMAG.

    Parameters
    ----------
    wav : `~astropy.units.quantity.Quantity`
        Quantity associated with values being converted
        (e.g., wavelength or frequency).

    vegaflux : `~astropy.units.quantity.Quantity`
        Flux of Vega at ``wav``.

    Returns
    -------
    eqv : list
        List of equivalencies.

    """
    vega_photlam = vegaflux.to(
        PHOTLAM, equivalencies=u.spectral_density(wav)).value

    def converter(x):
        """Set nan/inf to -99 mag."""
        val = -2.5 * np.log10(x / vega_photlam)
        result = np.zeros(val.shape, dtype=np.float64) - 99
        mask = np.isfinite(val)
        if result.ndim > 0:
            result[mask] = val[mask]
        elif mask:
            result = np.asarray(val)
        return result

    def iconverter(x):
        return vega_photlam * 10**(-0.4 * x)

    return [(PHOTLAM, VEGAMAG, converter, iconverter)]