def calc_riseset(t, target_name, location, prev_next, rise_set, horizon):
    """
    Time at next rise/set of ``target``.

    Parameters
    ----------
    t : `~astropy.time.Time` or other (see below)
        Time of observation. This will be passed in as the first argument to
        the `~astropy.time.Time` initializer, so it can be anything that
        `~astropy.time.Time` will accept (including a `~astropy.time.Time`
        object)

    target_name : str
        'moon' or 'sun'

    location : `~astropy.coordinates.EarthLocation`
        Observatory location

    prev_next : str - either 'previous' or 'next'
        Test next rise/set or previous rise/set

    rise_set : str - either 'rising' or 'setting'
        Compute prev/next rise or prev/next set

    location : `~astropy.coordinates.EarthLocation`
        Location of observer

    horizon : `~astropy.units.Quantity`
        Degrees above/below actual horizon to use
        for calculating rise/set times (i.e.,
        -6 deg horizon = civil twilight, etc.)

    Returns
    -------
    ret1 : `~astropy.time.Time`
        Time of rise/set
    """
    target = coord.get_body(target_name, t)
    t0 = _rise_set_trig(t, target, location, prev_next, rise_set)
    grid = t0 + np.linspace(-4*u.hour, 4*u.hour, 10)
    altaz_frame = coord.AltAz(obstime=grid, location=location)
    target = coord.get_body(target_name, grid)
    altaz = target.transform_to(altaz_frame)
    time_limits, altitude_limits = _horiz_cross(altaz.obstime, altaz.alt,
                                                rise_set, horizon)
    return _two_point_interp(time_limits, altitude_limits, horizon)