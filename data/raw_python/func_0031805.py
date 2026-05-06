def hav_dist(locs1, locs2):
    """
    Return a distance matrix between two set of coordinates.
    Use geometric distance (default) or haversine distance (if longlat=True).

    Parameters
    ----------
    locs1 : numpy.array
        The first set of coordinates as [(long, lat), (long, lat)].
    locs2 : numpy.array
        The second set of coordinates as [(long, lat), (long, lat)].

    Returns
    -------
    mat_dist : numpy.array
        The distance matrix between locs1 and locs2
    """
#    locs1 = np.radians(locs1)
#    locs2 = np.radians(locs2)
    cos_lat1 = np.cos(locs1[..., 0])
    cos_lat2 = np.cos(locs2[..., 0])
    cos_lat_d = np.cos(locs1[..., 0] - locs2[..., 0])
    cos_lon_d = np.cos(locs1[..., 1] - locs2[..., 1])
    return 6367000 * np.arccos(
        cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))