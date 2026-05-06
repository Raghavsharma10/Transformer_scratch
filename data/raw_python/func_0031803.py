def make_regular_points(bounds, resolution, longlat=True):
    """
    Return a regular grid of points within `bounds` with the specified
    resolution.

    Parameters
    ----------
    bounds : 4-floats tuple
        The bbox of the grid, as xmin, ymin, xmax, ymax.
    resolution : int
        The resolution to use, in the same unit as `bounds`

    Returns
    -------
    points : numpy.array
        An array of coordinates
    shape : 2-floats tuple
        The number of points on each dimension (width, height)
    """
#    xmin, ymin, xmax, ymax = bounds
    minlon, minlat, maxlon, maxlat = bounds
    offset_lon = (maxlon - minlon) / 8
    offset_lat = (maxlat - minlat) / 8
    minlon -= offset_lon
    maxlon += offset_lon
    minlat -= offset_lat
    maxlat += offset_lat

    if longlat:
        height = hav_dist(
                np.array([(maxlon + minlon) / 2, minlat]),
                np.array([(maxlon + minlon) / 2, maxlat])
                )
        width = hav_dist(
                np.array([minlon, (maxlat + minlat) / 2]),
                np.array([maxlon, (maxlat + minlat) / 2])
                )
    else:
        height = np.linalg.norm(
            np.array([(maxlon + minlon) / 2, minlat])
            - np.array([(maxlon + minlon) / 2, maxlat]))
        width = np.linalg.norm(
            np.array([minlon, (maxlat + minlat) / 2])
            - np.array([maxlon, (maxlat + minlat) / 2]))

    nb_x = int(round(width / resolution))
    nb_y = int(round(height / resolution))
    if nb_y * 0.6 > nb_x:
        nb_x = int(nb_x + nb_x / 3)
    elif nb_x * 0.6 > nb_y:
        nb_y = int(nb_y + nb_y / 3)
    return (
        np.linspace(minlon, maxlon, nb_x),
        np.linspace(minlat, maxlat, nb_y),
        (nb_y, nb_x)
        )