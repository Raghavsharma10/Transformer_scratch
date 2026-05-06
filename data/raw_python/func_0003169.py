def radar_xsect(scatterer, h_pol=True):
    """Radar cross section for the current setup.    

    Args:
        scatterer: a Scatterer instance.
        h_pol: If True (default), use horizontal polarization.
        If False, use vertical polarization.

    Returns:
        The radar cross section.
    """
    Z = scatterer.get_Z()
    if h_pol:
        return 2 * np.pi * \
            (Z[0,0] - Z[0,1] - Z[1,0] + Z[1,1])
    else:
        return 2 * np.pi * \
            (Z[0,0] + Z[0,1] + Z[1,0] + Z[1,1])