def prof_pressure(altitude, z_coef=(1.94170e-9, -5.14580e-7, 4.57018e-5,
                                    -1.55620e-3, -4.61994e-2, 2.99955)):
    """
    Return pressure for given altitude.

    This function evaluates a polynomial at altitudes values.

    Parameters
    ----------
    altitude : array-like
        altitude values [km].
    z_coef : array-like
        coefficients of the polynomial (default values are for the US
        Standard Atmosphere).

    Returns
    -------
    pressure : array-like
        pressure values [hPa] (same shape than the altitude input array).

    See Also
    --------
    prof_altitude : Returns altitude for
        given pressure.
    prof_temperature : Returns air temperature for
        given altitude.

    Notes
    -----
    Default coefficient values represent a 5th degree polynomial which had
    been fitted to USA data from 0-100 km. Accuracy is on the order of 1% for
    0-100 km and 0.5% below 30 km. This function, with default values, may thus
    produce bad results with altitude > 100 km.

    Examples
    --------
    >>> prof_pressure([0, 10, 20])
    array([ 998.96437334,  264.658697  ,   55.28114631])

    """
    altitude = np.asarray(altitude)
    pressure = np.power(10, np.polyval(z_coef, altitude.flatten()))
    return pressure.reshape(altitude.shape)