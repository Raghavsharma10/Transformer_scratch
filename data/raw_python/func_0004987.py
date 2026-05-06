def GaussSpheres(q, A, R0, sigma, N=1000, weighting='intensity'):
    """Scattering of a population of non-correlated spheres (radii from a gaussian distribution)

    Inputs:
    -------
        ``q``: independent variable
        ``A``: scaling factor
        ``R0``: expectation of ``R``
        ``sigma``: hwhm of ``R``
        ``weighting``: 'intensity' (default), 'volume' or 'number'

    Non-fittable inputs:
    --------------------
        ``N``: the (integer) number of spheres

    Formula:
    --------
        The integral of ``F_sphere^2(q,R) * P(R)`` where ``P(R)`` is a
        gaussian (normal) distribution of the radii.

    """
    Rmin = max(0, R0 - 3 * sigma)
    Rmax = R0 + 3 * sigma
    R = np.linspace(Rmin, Rmax, N + 1)[1:]
    P = 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(R - R0) ** 2 / (2 * sigma ** 2))
    def Fsphere_outer(q, R):
        qR = np.outer(q, R)
        return 3 / qR ** 3 * (np.sin(qR) - qR * np.cos(qR))
    V=R**3*4*np.pi/3.
    if weighting=='intensity':
        P=P*V*V
    elif weighting=='volume':
        P=P*V
    elif weighting=='number':
        pass
    else:
        raise ValueError('Invalid weighting: '+str(weighting))    
    I = (Fsphere_outer(q, R) ** 2 * np.outer(np.ones_like(q), P))
    return A * I.sum(1) / P.sum()