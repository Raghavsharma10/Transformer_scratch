def _astropy_time_from_LST(t, LST, location, prev_next):
    """
    Convert a Local Sidereal Time to an astropy Time object.

    The local time is related to the LST through the RA of the Sun.
    This routine uses this relationship to convert a LST to an astropy
    time object.

    Returns
    -------
    ret1 : `~astropy.time.Time`
        time corresponding to LST
    """
    # now we need to figure out time to return from LST
    raSun = coord.get_sun(t).ra

    # calculate Greenwich Apparent Solar Time, which we will use as ~UTC for now
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # ignore astropy deprecation warnings
        lon = location.longitude
    solarTime = LST - raSun + 12*u.hourangle - lon

    # assume this is on the same day as supplied time, and fix later
    first_guess = Time(
        u.d*int(t.mjd) + u.hour*solarTime.wrap_at('360d').hour,
        format='mjd'
    )

    # Equation of time is difference between GAST and UTC
    eot = _equation_of_time(first_guess)
    first_guess = first_guess - u.hour * eot.value

    if prev_next == 'next':
        # if 'next', we want time to be greater than given time
        mask = first_guess < t
        rise_set_time = first_guess + mask * u.sday
    else:
        # if 'previous', we want time to be less than given time
        mask = first_guess > t
        rise_set_time = first_guess - mask * u.sday
    return rise_set_time