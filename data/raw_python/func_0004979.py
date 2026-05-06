def Fsphere(q, R):
    """Scattering form-factor amplitude of a sphere normalized to F(q=0)=V

    Inputs:
    -------
        ``q``: independent variable
        ``R``: sphere radius

    Formula:
    --------
        ``4*pi/q^3 * (sin(qR) - qR*cos(qR))``
    """
    return 4 * np.pi / q ** 3 * (np.sin(q * R) - q * R * np.cos(q * R))