def make_regular_points_with_no_res(bounds, nb_points=10000):
    """
    Return a regular grid of points within `bounds` with the specified
    number of points (or a close approximate value).

    Parameters
    ----------
    bounds : 4-floats tuple
        The bbox of the grid, as xmin, ymin, xmax, ymax.
    nb_points : int, optionnal
        The desired number of points (default: 10000)

    Returns
    -------
    points : numpy.array
        An array of coordinates
    shape : 2-floats tuple
        The number of points on each dimension (width, height)
    """
    minlon, minlat, maxlon, maxlat = bounds
    minlon, minlat, maxlon, maxlat = bounds
    offset_lon = (maxlon - minlon) / 8
    offset_lat = (maxlat - minlat) / 8
    minlon -= offset_lon
    maxlon += offset_lon
    minlat -= offset_lat
    maxlat += offset_lat

    nb_x = int(nb_points**0.5)
    nb_y = int(nb_points**0.5)

    return (
        np.linspace(minlon, maxlon, nb_x),
        np.linspace(minlat, maxlat, nb_y),
        (nb_y, nb_x)
        )