def ext_xsect(scatterer, h_pol=True):
    """Extinction cross section for the current setup, with polarization.    

    Args:
        scatterer: a Scatterer instance.
        h_pol: If True (default), use horizontal polarization.
        If False, use vertical polarization.

    Returns:
        The extinction cross section.
    """

    if scatterer.psd_integrator is not None:
        try:
            return scatterer.psd_integrator.get_angular_integrated(
                scatterer.psd, scatterer.get_geometry(), "ext_xsect")
        except AttributeError:
            # Fall back to the usual method of computing this from S
            pass

    old_geom = scatterer.get_geometry()
    (thet0, thet, phi0, phi, alpha, beta) = old_geom
    try:
        scatterer.set_geometry((thet0, thet0, phi0, phi0, alpha, beta))
        S = scatterer.get_S()        
    finally:
        scatterer.set_geometry(old_geom)



    if h_pol:
        return 2 * scatterer.wavelength * S[1,1].imag
    else:
        return 2 * scatterer.wavelength * S[0,0].imag