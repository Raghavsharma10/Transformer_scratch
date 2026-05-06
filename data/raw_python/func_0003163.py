def sca_intensity(scatterer, h_pol=True):
    """Scattering intensity (phase function) for the current setup.    

    Args:
        scatterer: a Scatterer instance.
        h_pol: If True (default), use horizontal polarization.
        If False, use vertical polarization.

    Returns:
        The differential scattering cross section.
    """
    Z = scatterer.get_Z()
    return (Z[0,0] - Z[0,1]) if h_pol else (Z[0,0] + Z[0,1])