def prof_altitude(pressure, p_coef=(-0.028389, -0.0493698, 0.485718, 0.278656,
                                    -17.5703, 48.0926)):
    """
    Return altitude for given pressure.

    This function evaluates a polynomial at log10(pressure) values.

    Parameters
    ----------
    pressure : array-like
        pressure values [hPa].
    p_coef : array-like
        coefficients of the polynomial (default values are for the US
        Standard Atmosphere).

    Returns
    -------
    altitude : array-like
        altitude values [km] (same shape than the pressure input array).

    See Also
    --------
    prof_pressure : Returns pressure for
        given altitude.
    prof_temperature : Returns air temperature for
        given altitude.

    Notes
    -----
    Default coefficient values represent a 5th degree polynomial which had
    been fitted to USSA data from 0-100 km. Accuracy is on the order of 1% for
    0-100 km and 0.5% below 30 km. This function, with default values, may thus
    produce bad results with pressure less than about 3e-4 hPa.

    Examples
    --------
    >>> prof_altitude([1000, 800, 600])
    array([ 0.1065092 ,  1.95627858,  4.2060627 ])

    """
    pressure = np.asarray(pressure)
    altitude = np.polyval(p_coef, np.log10(pressure.flatten()))
    return altitude.reshape(pressure.shape)