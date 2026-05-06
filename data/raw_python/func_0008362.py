def _two_point_interp(times, altitudes, horizon=0*u.deg):
    """
    Do linear interpolation between two ``altitudes`` at
    two ``times`` to determine the time where the altitude
    goes through zero.

    Parameters
    ----------
    times : `~astropy.time.Time`
        Two times for linear interpolation between

    altitudes : array of `~astropy.units.Quantity`
        Two altitudes for linear interpolation between

    horizon : `~astropy.units.Quantity`
        Solve for the time when the altitude is equal to
        reference_alt.

    Returns
    -------
    t : `~astropy.time.Time`
        Time when target crosses the horizon

    """
    if not isinstance(times, Time):
        return MAGIC_TIME
    else:
        slope = (altitudes[1] - altitudes[0])/(times[1].jd - times[0].jd)
        return Time(times[1].jd - ((altitudes[1] - horizon)/slope).value,
                    format='jd')