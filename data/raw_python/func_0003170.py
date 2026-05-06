def refl(scatterer, h_pol=True):
    """Reflectivity (with number concentration N=1) for the current setup.

    Args:
        scatterer: a Scatterer instance.
        h_pol: If True (default), use horizontal polarization.
        If False, use vertical polarization.

    Returns:
        The reflectivity.

    NOTE: To compute reflectivity in dBZ, give the particle diameter and
    wavelength in [mm], then take 10*log10(Zi).
    """
    return scatterer.wavelength**4/(np.pi**5*scatterer.Kw_sqr) * \
        radar_xsect(scatterer, h_pol)