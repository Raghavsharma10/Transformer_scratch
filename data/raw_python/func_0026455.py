def checklat(lat, name='lat'):
    """Makes sure the latitude is inside [-90, 90], clipping close values
    (tolerance 1e-4).

    Parameters
    ==========
    lat : array_like
        latitude
    name : str, optional
        parameter name to use in the exception message

    Returns
    =======
    lat : ndarray or float
        Same as input where values just outside the range have been
        clipped to [-90, 90]

    Raises
    ======
    ValueError
        if any values are too far outside the range [-90, 90]
    """

    if np.all(np.float64(lat) >= -90) and np.all(np.float64(lat) <= 90):
        return lat

    if np.isscalar(lat):
        if lat > 90 and np.isclose(lat, 90, rtol=0, atol=1e-4):
            lat = 90
            return lat
        elif lat < -90 and np.isclose(lat, -90, rtol=0, atol=1e-4):
            lat = -90
            return lat
    else:
        lat = np.float64(lat)  # make sure we have an array, not list
        lat[(lat > 90) & (np.isclose(lat, 90, rtol=0, atol=1e-4))] = 90
        lat[(lat < -90) & (np.isclose(lat, -90, rtol=0, atol=1e-4))] = -90
        if np.all(lat >= -90) and np.all(lat <= 90):
            return lat

    # we haven't returned yet, so raise exception
    raise ValueError(name + ' must be in [-90, 90]')