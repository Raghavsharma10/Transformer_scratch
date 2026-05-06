def _horiz_cross(t, alt, rise_set, horizon=0*u.degree):
    """
    Find time ``t`` when values in array ``a`` go from
    negative to positive or positive to negative (exclude endpoints)

    ``return_limits`` will return nearest times to zero-crossing.

    Parameters
    ----------
    t : `~astropy.time.Time`
        Grid of times
    alt : `~astropy.units.Quantity`
        Grid of altitudes
    rise_set : {"rising",  "setting"}
        Calculate either rising or setting across the horizon
    horizon : float
        Number of degrees above/below actual horizon to use
        for calculating rise/set times (i.e.,
        -6 deg horizon = civil twilight, etc.)

    Returns
    -------
    Returns the lower and upper limits on the time and altitudes
    of the horizon crossing.
    """
    if rise_set == 'rising':
        # Find index where altitude goes from below to above horizon
        condition = (alt[:-1] < horizon) * (alt[1:] > horizon)
    elif rise_set == 'setting':
        # Find index where altitude goes from above to below horizon
        condition = (alt[:-1] > horizon) * (alt[1:] < horizon)

    if np.count_nonzero(condition) == 0:
        warnmsg = ('Target does not cross horizon={} within '
                   '8 hours of trigonometric estimate'.format(horizon))
        warnings.warn(warnmsg)

        # Fill in missing time with MAGIC_TIME
        time_inds = np.nan
        times = [np.nan, np.nan]
        altitudes = [np.nan, np.nan]
    else:
        time_inds = np.nonzero(condition)[0][0]
        times = t[time_inds:time_inds+2]
        altitudes = alt[time_inds:time_inds+2]

    return times, altitudes