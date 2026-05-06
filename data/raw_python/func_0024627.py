def spectral_density_count(wav, area):
    """Flux equivalencies between PHOTLAM and count/OBMAG.

    Parameters
    ----------
    wav : `~astropy.units.quantity.Quantity`
        Quantity associated with values being converted
        (e.g., wavelength or frequency).

    area : `~astropy.units.quantity.Quantity`
        Telescope collecting area.

    Returns
    -------
    eqv : list
        List of equivalencies.

    """
    from .binning import calculate_bin_widths, calculate_bin_edges

    wav = wav.to(u.AA, equivalencies=u.spectral())
    area = area.to(AREA)
    bin_widths = calculate_bin_widths(calculate_bin_edges(wav))
    factor = bin_widths.value * area.value

    def converter_count(x):
        return x * factor

    def iconverter_count(x):
        return x / factor

    def converter_obmag(x):
        return -2.5 * np.log10(x * factor)

    def iconverter_obmag(x):
        return 10**(-0.4 * x) / factor

    return [(PHOTLAM, u.count, converter_count, iconverter_count),
            (PHOTLAM, OBMAG, converter_obmag, iconverter_obmag)]