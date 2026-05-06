def errors(source, model, wcshelper):
    """
    Convert pixel based errors into sky coord errors

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
    # copy the errors from the model
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

    source.err_peak_flux = err_amp
    pix_errs = [err_xo, err_yo, err_sx, err_sy, err_theta]

    log.debug("Pix errs: {0}".format(pix_errs))

    ref = wcshelper.pix2sky([xo, yo])
    # check to see if the reference position has a valid WCS coordinate
    # It is possible for this to fail, even if the ra/dec conversion works elsewhere
    if not all(np.isfinite(ref)):
        source.flags |= flags.WCSERR
        source.err_peak_flux = source.err_a = source.err_b = source.err_pa = ERR_MASK
        source.err_ra = source.err_dec = source.err_int_flux = ERR_MASK
        return source

    # position errors
    if model[prefix + 'xo'].vary and model[prefix + 'yo'].vary \
            and all(np.isfinite([err_xo, err_yo])):
        offset = wcshelper.pix2sky([xo + err_xo, yo + err_yo])
        source.err_ra = gcd(ref[0], ref[1], offset[0], ref[1])
        source.err_dec = gcd(ref[0], ref[1], ref[0], offset[1])
    else:
        source.err_ra = source.err_dec = -1

    if model[prefix + 'theta'].vary and np.isfinite(err_theta):
        # pa error
        off1 = wcshelper.pix2sky([xo + sx * np.cos(np.radians(theta)), yo + sy * np.sin(np.radians(theta))])
        off2 = wcshelper.pix2sky(
            [xo + sx * np.cos(np.radians(theta + err_theta)), yo + sy * np.sin(np.radians(theta + err_theta))])
        source.err_pa = abs(bear(ref[0], ref[1], off1[0], off1[1]) - bear(ref[0], ref[1], off2[0], off2[1]))
    else:
        source.err_pa = ERR_MASK

    if model[prefix + 'sx'].vary and model[prefix + 'sy'].vary \
            and all(np.isfinite([err_sx, err_sy])):
        # major axis error
        ref = wcshelper.pix2sky([xo + sx * np.cos(np.radians(theta)), yo + sy * np.sin(np.radians(theta))])
        offset = wcshelper.pix2sky(
            [xo + (sx + err_sx) * np.cos(np.radians(theta)), yo + sy * np.sin(np.radians(theta))])
        source.err_a = gcd(ref[0], ref[1], offset[0], offset[1]) * 3600

        # minor axis error
        ref = wcshelper.pix2sky([xo + sx * np.cos(np.radians(theta + 90)), yo + sy * np.sin(np.radians(theta + 90))])
        offset = wcshelper.pix2sky(
            [xo + sx * np.cos(np.radians(theta + 90)), yo + (sy + err_sy) * np.sin(np.radians(theta + 90))])
        source.err_b = gcd(ref[0], ref[1], offset[0], offset[1]) * 3600
    else:
        source.err_a = source.err_b = ERR_MASK

    sqerr = 0
    sqerr += (source.err_peak_flux / source.peak_flux) ** 2 if source.err_peak_flux > 0 else 0
    sqerr += (source.err_a / source.a) ** 2 if source.err_a > 0 else 0
    sqerr += (source.err_b / source.b) ** 2 if source.err_b > 0 else 0
    if sqerr == 0:
        source.err_int_flux = ERR_MASK
    else:
        source.err_int_flux = abs(source.int_flux * np.sqrt(sqerr))

    return source