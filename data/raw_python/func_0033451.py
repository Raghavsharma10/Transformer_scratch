def enu_to_ecef_vector(east, north, up, glat, glong):
    """Converts vector from East, North, Up components to ECEF
    
    Position of vector in geospace may be specified in either
    geocentric or geodetic coordinates, with corresponding expression
    of the vector using radial or ellipsoidal unit vectors.
    
    Parameters
    ----------
    east : float or array-like
        Eastward component of vector
    north : float or array-like
        Northward component of vector
    up : float or array-like
        Upward component of vector
    latitude : float or array_like
        Geodetic or geocentric latitude (degrees)
    longitude : float or array_like
        Geodetic or geocentric longitude (degrees)
    
    Returns
    -------
    x, y, z
        Vector components along ECEF x, y, and z directions
    
    """
    
    # convert lat and lon in degrees to radians
    rlat = np.radians(glat)
    rlon = np.radians(glong)
    
    x = -east*np.sin(rlon) - north*np.cos(rlon)*np.sin(rlat) + up*np.cos(rlon)*np.cos(rlat)
    y = east*np.cos(rlon) - north*np.sin(rlon)*np.sin(rlat) + up*np.sin(rlon)*np.cos(rlat)
    z = north*np.cos(rlat) + up*np.sin(rlat)

    return x, y, z