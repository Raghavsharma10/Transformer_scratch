def distance(p0, p1, deg=True, r=r_earth_mean):
    """
    Return the distance between two points on the surface of the Earth.

    Parameters
    ----------
    p0 : point-like (or array of point-like) [longitude, latitude] objects
    p1 : point-like (or array of point-like) [longitude, latitude] objects
    deg : bool, optional (default True)
        indicates if p0 and p1 are specified in degrees 
    r : float, optional (default r_earth_mean)
        radius of the sphere 

    Returns
    -------
    d : float

    Reference
    ---------
    http://www.movable-type.co.uk/scripts/latlong.html - Distance

    Note: Spherical earth model. By default uses radius of 6371.0 km.

    """
    single, (p0, p1) = _to_arrays((p0, 2), (p1, 2))
    if deg:
        p0 = np.radians(p0)
        p1 = np.radians(p1)

    lon0, lat0 = p0[:,0], p0[:,1]
    lon1, lat1 = p1[:,0], p1[:,1]

    # h_x used to denote haversine(x): sin^2(x / 2)
    h_dlat = sin((lat1 - lat0) / 2.0) ** 2
    h_dlon = sin((lon1 - lon0) / 2.0) ** 2
    h_angle = h_dlat + cos(lat0) * cos(lat1) * h_dlon
    angle = 2.0 * arcsin(sqrt(h_angle))
    d = r * angle

    if single:
        d = d[0]

    return d