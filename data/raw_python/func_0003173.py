def Kdp(scatterer):
    """
    Specific differential phase (K_dp) for the current setup.

    Args:
        scatterer: a Scatterer instance.

    Returns:
        K_dp [deg/km].

    NOTE: This only returns the correct value if the particle diameter and
    wavelength are given in [mm]. The scatterer object should be set to 
    forward scattering geometry before calling this function.
    """
    if (scatterer.thet0 != scatterer.thet) or \
        (scatterer.phi0 != scatterer.phi):
        
        raise ValueError("A forward scattering geometry is needed to " + \
            "compute the specific differential phase.")

    S = scatterer.get_S()
    return 1e-3 * (180.0/np.pi) * scatterer.wavelength * (S[1,1]-S[0,0]).real