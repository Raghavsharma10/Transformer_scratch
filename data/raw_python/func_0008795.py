def condon_errors(source, theta_n, psf=None):
    """
    Calculate the parameter errors for a fitted source
    using the description of Condon'97
    All parameters are assigned errors, assuming that all params were fit.
    If some params were held fixed then these errors are overestimated.

    Parameters
    ----------
    source : :class:`AegeanTools.models.SimpleSource`
        The source which was fit.

    theta_n : float or None
        A measure of the beam sampling. (See Condon'97).

    psf : :class:`AegeanTools.fits_image.Beam`
        The psf at the location of the source.

    Returns
    -------
    None

    """

    # indices for the calculation or rho
    alphas = {'amp': (3. / 2, 3. / 2),
              'major': (5. / 2, 1. / 2),
              'xo': (5. / 2, 1. / 2),
              'minor': (1. / 2, 5. / 2),
              'yo': (1. / 2, 5. / 2),
              'pa': (1. / 2, 5. / 2)}

    major = source.a / 3600.  # degrees
    minor = source.b / 3600.  # degrees
    phi = np.radians(source.pa)  # radians
    if psf is not None:
        beam = psf.get_beam(source.ra, source.dec)
        if beam is not None:
            theta_n = np.hypot(beam.a, beam.b)
            print(beam, theta_n)

    if theta_n is None:
        source.err_a = source.err_b = source.err_peak_flux = source.err_pa = source.err_int_flux = 0.0
        return

    smoothing = major * minor / (theta_n ** 2)
    factor1 = (1 + (major / theta_n))
    factor2 = (1 + (minor / theta_n))
    snr = source.peak_flux / source.local_rms
    # calculation of rho2 depends on the parameter being used so we lambda this into a function
    rho2 = lambda x: smoothing / 4 * factor1 ** alphas[x][0] * factor2 ** alphas[x][1] * snr ** 2

    source.err_peak_flux = source.peak_flux * np.sqrt(2 / rho2('amp'))
    source.err_a = major * np.sqrt(2 / rho2('major')) * 3600.  # arcsec
    source.err_b = minor * np.sqrt(2 / rho2('minor')) * 3600.  # arcsec

    err_xo2 = 2. / rho2('xo') * major ** 2 / (8 * np.log(2))  # Condon'97 eq 21
    err_yo2 = 2. / rho2('yo') * minor ** 2 / (8 * np.log(2))
    source.err_ra = np.sqrt(err_xo2 * np.sin(phi)**2 + err_yo2 * np.cos(phi)**2)
    source.err_dec = np.sqrt(err_xo2 * np.cos(phi)**2 + err_yo2 * np.sin(phi)**2)

    if (major == 0) or (minor == 0):
        source.err_pa = ERR_MASK
    # if major/minor are very similar then we should not be able to figure out what pa is.
    elif abs(2 * (major-minor) / (major+minor)) < 0.01:
        source.err_pa = ERR_MASK
    else:
        source.err_pa = np.degrees(np.sqrt(4 / rho2('pa')) * (major * minor / (major ** 2 - minor ** 2)))

    # integrated flux error
    err2 = (source.err_peak_flux / source.peak_flux) ** 2
    err2 += (theta_n ** 2 / (major * minor)) * ((source.err_a / source.a) ** 2 + (source.err_b / source.b) ** 2)
    source.err_int_flux = source.int_flux * np.sqrt(err2)
    return