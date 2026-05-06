def ssa(scatterer, h_pol=True):
    """Single-scattering albedo for the current setup, with polarization.    

    Args:
        scatterer: a Scatterer instance.
        h_pol: If True (default), use horizontal polarization.
        If False, use vertical polarization.

    Returns:
        The single-scattering albedo.
    """

    ext_xs = ext_xsect(scatterer, h_pol=h_pol)
    return sca_xsect(scatterer, h_pol=h_pol)/ext_xs if ext_xs > 0.0 else 0.0