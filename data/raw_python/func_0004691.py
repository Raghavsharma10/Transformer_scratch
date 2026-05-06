def propagate(p0, angle, d, deg=True, bearing=False, r=r_earth_mean):
    """
    Given an initial point and angle, move distance d along the surface

    Parameters
    ----------
    p0 : point-like (or array of point-like) [lon, lat] objects
    angle : float (or array of float)
        bearing. Note that by default, 0 degrees is due East increasing 
        clockwise so that 90 degrees is due North. See the bearing flag
        to change the meaning of this angle
    d : float (or array of float)
        distance to move. The units of d should be consistent with input r
    deg : bool, optional (default True)
        Whether both p0 and angle are specified in degrees. The output
        points will also match the value of this flag.
    bearing : bool, optional (default False)
        Indicates whether to interpret the input angle as the classical
        definition of bearing.
    r : float, optional (default r_earth_mean)
        radius of the sphere


    Reference
    ---------
    http://www.movable-type.co.uk/scripts/latlong.html - Destination

    Note: Spherical earth model. By default uses radius of 6371.0 km.

    """
    single, (p0, angle, d) = _to_arrays((p0, 2), (angle, 1), (d, 1))
    if deg:
        p0 = np.radians(p0)
        angle = np.radians(angle)

    if not bearing:
        angle = np.pi / 2.0 - angle

    lon0, lat0 = p0[:,0], p0[:,1]

    angd = d / r
    lat1 = arcsin(sin(lat0) * cos(angd) + cos(lat0) * sin(angd) * cos(angle))

    a = sin(angle) * sin(angd) * cos(lat0)
    b = cos(angd) - sin(lat0) * sin(lat1)
    lon1 = lon0 + arctan2(a, b)

    p1 = np.column_stack([lon1, lat1])

    if deg:
        p1 = np.degrees(p1)

    if single:
        p1 = p1[0]

    return p1