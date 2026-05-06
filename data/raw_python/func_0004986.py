def LogNormSpheres(q, A, mu, sigma, N=1000):
    """Scattering of a population of non-correlated spheres (radii from a log-normal distribution)

    Inputs:
    -------
        ``q``: independent variable
        ``A``: scaling factor
        ``mu``: expectation of ``ln(R)``
        ``sigma``: hwhm of ``ln(R)``

    Non-fittable inputs:
    --------------------
        ``N``: the (integer) number of spheres

    Formula:
    --------
        The integral of ``F_sphere^2(q,R) * P(R)`` where ``P(R)`` is a
        log-normal distribution of the radii.

    """
    Rmin = 0
    Rmax = np.exp(mu + 3 * sigma)
    R = np.linspace(Rmin, Rmax, N + 1)[1:]
    P = 1 / np.sqrt(2 * np.pi * sigma ** 2 * R ** 2) * np.exp(-(np.log(R) - mu) ** 2 / (2 * sigma ** 2))
    def Fsphere_outer(q, R):
        qR = np.outer(q, R)
        q1 = np.outer(q, np.ones_like(R))
        return 4 * np.pi / q1 ** 3 * (np.sin(qR) - qR * np.cos(qR))
    I = (Fsphere_outer(q, R) ** 2 * np.outer(np.ones_like(q), P))
    return A * I.sum(1) / P.sum()