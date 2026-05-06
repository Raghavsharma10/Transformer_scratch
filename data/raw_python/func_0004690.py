def course(p0, p1, deg=True, bearing=False):
    """
    Compute the initial bearing along the great circle from p0 to p1

    NB: The angle returned by course() is not the traditional definition of
    bearing. It is definted such that 0 degrees to due East increasing
    counter-clockwise such that 90 degrees is due North. To obtain the bearing
    (0 degrees is due North increasing clockwise so that 90 degrees is due
    East), set the bearing flag input to True.

    Parameters
    ----------
    p0 : point-like (or array of point-like) [lon, lat] objects
    p1 : point-like (or array of point-like) [lon, lat] objects
    deg : bool, optional (default True)
        indicates if p0 and p1 are specified in degrees. The returned
        angle is returned in the same units as the input.
    bearing : bool, optional (default False)
        If True, use the classical definition of bearing where 0 degrees is
        due North increasing clockwise so that and 90 degrees is due East.

    Reference
    ---------
    http://www.movable-type.co.uk/scripts/latlong.html - Bearing

    """
    single, (p0, p1) = _to_arrays((p0, 2), (p1, 2))
    if deg:
        p0 = np.radians(p0)
        p1 = np.radians(p1)

    lon0, lat0 = p0[:,0], p0[:,1]
    lon1, lat1 = p1[:,0], p1[:,1]

    dlon = lon1 - lon0
    a = sin(dlon) * cos(lat1)
    b = cos(lat0) * sin(lat1) - sin(lat0) * cos(lat1) * cos(dlon)

    if bearing:
        angle = arctan2(a, b)
    else:
        angle = arctan2(b, a)

    if deg:
        angle = np.degrees(angle)

    if single:
        angle = angle[0]

    return angle