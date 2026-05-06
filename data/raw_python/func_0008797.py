def new_errors(source, model, wcshelper):  # pragma: no cover
    """
    Convert pixel based errors into sky coord errors
    Uses covariance matrix for ra/dec errors
    and calculus approach to a/b/pa errors

    Parameters
    ----------
    source : :class:`AegeanTools.models.SimpleSource`
        The source which was fit.

    model : lmfit.Parameters
        The model which was fit.

    wcshelper : :class:`AegeanTools.wcs_helpers.WCSHelper`
        WCS information.

    Returns
    -------
    source : :class:`AegeanTools.models.SimpleSource`
        The modified source obejct.

    """

    # if the source wasn't fit then all errors are -1
    if source.flags & (flags.NOTFIT | flags.FITERR):
        source.err_peak_flux = source.err_a = source.err_b = source.err_pa = ERR_MASK
        source.err_ra = source.err_dec = source.err_int_flux = ERR_MASK
        return source
    # copy the errors/values from the model
    prefix = "c{0}_".format(source.source)
    err_amp = model[prefix + 'amp'].stderr
    xo, yo = model[prefix + 'xo'].value, model[prefix + 'yo'].value
    err_xo = model[prefix + 'xo'].stderr
    err_yo = model[prefix + 'yo'].stderr

    sx, sy = model[prefix + 'sx'].value, model[prefix + 'sy'].value
    err_sx = model[prefix + 'sx'].stderr
    err_sy = model[prefix + 'sy'].stderr

    theta = model[prefix + 'theta'].value
    err_theta = model[prefix + 'theta'].stderr

    # the peak flux error doesn't need to be converted, just copied
    source.err_peak_flux = err_amp

    pix_errs = [err_xo, err_yo, err_sx, err_sy, err_theta]

    # check for inf/nan errors -> these sources have poor fits.
    if not all(a is not None and np.isfinite(a) for a in pix_errs):
        source.flags |= flags.FITERR
        source.err_peak_flux = source.err_a = source.err_b = source.err_pa = ERR_MASK
        source.err_ra = source.err_dec = source.err_int_flux = ERR_MASK
        return source

    # calculate the reference coordinate
    ref = wcshelper.pix2sky([xo, yo])
    # check to see if the reference position has a valid WCS coordinate
    # It is possible for this to fail, even if the ra/dec conversion works elsewhere
    if not all(np.isfinite(ref)):
        source.flags |= flags.WCSERR
        source.err_peak_flux = source.err_a = source.err_b = source.err_pa = ERR_MASK
        source.err_ra = source.err_dec = source.err_int_flux = ERR_MASK
        return source

    # calculate position errors by transforming the error ellipse
    if model[prefix + 'xo'].vary and model[prefix + 'yo'].vary:
        # determine the error ellipse from the Jacobian
        mat = model.covar[1:3, 1:3]
        if not(np.all(np.isfinite(mat))):
            source.err_ra = source.err_dec = ERR_MASK
        else:
            (a, b), e = np.linalg.eig(mat)
            pa = np.degrees(np.arctan2(*e[0]))
            # transform this ellipse into sky coordinates
            _, _, major, minor, pa = wcshelper.pix2sky_ellipse([xo, yo], a, b, pa)

            # now determine the radius of the ellipse along the ra/dec directions.
            source.err_ra = major*minor / np.hypot(major*np.sin(np.radians(pa)), minor*np.cos(np.radians(pa)))
            source.err_dec = major*minor / np.hypot(major*np.cos(np.radians(pa)), minor*np.sin(np.radians(pa)))
    else:
        source.err_ra = source.err_dec = -1

    if model[prefix + 'theta'].vary:
        # pa error
        off1 = wcshelper.pix2sky([xo + sx * np.cos(np.radians(theta)), yo + sy * np.sin(np.radians(theta))])
        # offset by 1 degree
        off2 = wcshelper.pix2sky(
            [xo + sx * np.cos(np.radians(theta + 1)), yo + sy * np.sin(np.radians(theta + 1))])
        # scale the initial theta error by this amount
        source.err_pa = abs(bear(ref[0], ref[1], off1[0], off1[1]) - bear(ref[0], ref[1], off2[0], off2[1])) * err_theta
    else:
        source.err_pa = ERR_MASK

    if model[prefix + 'sx'].vary and model[prefix + 'sy'].vary:
        # major axis error
        ref = wcshelper.pix2sky([xo + sx * np.cos(np.radians(theta)), yo + sy * np.sin(np.radians(theta))])
        # offset by 0.1 pixels
        offset = wcshelper.pix2sky(
            [xo + (sx + 0.1) * np.cos(np.radians(theta)), yo + sy * np.sin(np.radians(theta))])
        source.err_a = gcd(ref[0], ref[1], offset[0], offset[1])/0.1 * err_sx * 3600

        # minor axis error
        ref = wcshelper.pix2sky([xo + sx * np.cos(np.radians(theta + 90)), yo + sy * np.sin(np.radians(theta + 90))])
        # offset by 0.1 pixels
        offset = wcshelper.pix2sky(
            [xo + sx * np.cos(np.radians(theta + 90)), yo + (sy + 0.1) * np.sin(np.radians(theta + 90))])
        source.err_b = gcd(ref[0], ref[1], offset[0], offset[1])/0.1*err_sy * 3600
    else:
        source.err_a = source.err_b = ERR_MASK
    sqerr = 0
    sqerr += (source.err_peak_flux / source.peak_flux) ** 2 if source.err_peak_flux > 0 else 0
    sqerr += (source.err_a / source.a) ** 2 if source.err_a > 0 else 0
    sqerr += (source.err_b / source.b) ** 2 if source.err_b > 0 else 0
    source.err_int_flux = abs(source.int_flux * np.sqrt(sqerr))

    return source