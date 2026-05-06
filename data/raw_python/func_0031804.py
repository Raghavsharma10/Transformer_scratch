def make_dist_mat(xy1, xy2, longlat=True):
    """
    Return a distance matrix between two set of coordinates.
    Use geometric distance (default) or haversine distance (if longlat=True).

    Parameters
    ----------
    xy1 : numpy.array
        The first set of coordinates as [(x, y), (x, y), (x, y)].
    xy2 : numpy.array
        The second set of coordinates as [(x, y), (x, y), (x, y)].
    longlat : boolean, optionnal
        Whether the coordinates are in geographic (longitude/latitude) format
        or not (default: False)

    Returns
    -------
    mat_dist : numpy.array
        The distance matrix between xy1 and xy2
    """
    if longlat:
        return hav_dist(xy1[:, None], xy2)
    else:
        d0 = np.subtract.outer(xy1[:, 0], xy2[:, 0])
        d1 = np.subtract.outer(xy1[:, 1], xy2[:, 1])
        return np.hypot(d0, d1)