def _rise_set_trig(t, target, location, prev_next, rise_set):
    """
    Crude time at next rise/set of ``target`` using spherical trig.

    This method is ~15 times faster than `_calcriseset`,
    and inherently does *not* take the atmosphere into account.

    The time returned should not be used in calculations; the purpose
    of this routine is to supply a guess to `_calcriseset`.

    Parameters
    ----------
    t : `~astropy.time.Time` or other (see below)
        Time of observation. This will be passed in as the first argument to
        the `~astropy.time.Time` initializer, so it can be anything that
        `~astropy.time.Time` will accept (including a `~astropy.time.Time`
        object)

    target : `~astropy.coordinates.SkyCoord`
        Position of target or multiple positions of that target
        at multiple times (if target moves, like the Sun)

    location : `~astropy.coordinates.EarthLocation`
        Observatory location

    prev_next : str - either 'previous' or 'next'
        Test next rise/set or previous rise/set

    rise_set : str - either 'rising' or 'setting'
        Compute prev/next rise or prev/next set

    Returns
    -------
    ret1 : `~astropy.time.Time`
        Time of rise/set
    """
    dec = target.transform_to(coord.ICRS).dec
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # ignore astropy deprecation warnings
        lat = location.latitude
    cosHA = -np.tan(dec)*np.tan(lat.radian)
    # find the absolute value of the hour Angle
    HA = coord.Longitude(np.fabs(np.arccos(cosHA)))
    # if rise, HA is -ve and vice versa
    if rise_set == 'rising':
        HA = -HA
    # LST = HA + RA
    LST = HA + target.ra

    return _astropy_time_from_LST(t, LST, location, prev_next)