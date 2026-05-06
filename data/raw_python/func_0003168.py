def asym(scatterer, h_pol=True):
    """Asymmetry parameter for the current setup, with polarization.    

    Args:
        scatterer: a Scatterer instance.
        h_pol: If True (default), use horizontal polarization.
        If False, use vertical polarization.

    Returns:
        The asymmetry parameter.
    """

    if scatterer.psd_integrator is not None:
        return scatterer.psd_integrator.get_angular_integrated(
            scatterer.psd, scatterer.get_geometry(), "asym")

    old_geom = scatterer.get_geometry()

    cos_t0 = np.cos(scatterer.thet0 * deg_to_rad)
    sin_t0 = np.sin(scatterer.thet0 * deg_to_rad)
    p0 = scatterer.phi0 * deg_to_rad
    def integrand(thet, phi):
        (scatterer.phi, scatterer.thet) = (phi*rad_to_deg, thet*rad_to_deg)
        cos_T_sin_t = 0.5 * (np.sin(2*thet)*cos_t0 + \
            (1-np.cos(2*thet))*sin_t0*np.cos(p0-phi))
        I = sca_intensity(scatterer, h_pol)
        return I * cos_T_sin_t

    try:
        cos_int = dblquad(integrand, 0.0, 2*np.pi, lambda x: 0.0, 
            lambda x: np.pi)[0]
    finally:
        scatterer.set_geometry(old_geom)

    return cos_int/sca_xsect(scatterer, h_pol)