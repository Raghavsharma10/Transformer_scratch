def geocentric_to_ecef(latitude, longitude, altitude):
    """Convert geocentric coordinates into ECEF
    
    Parameters
    ----------
    latitude : float or array_like
        Geocentric latitude (degrees)
    longitude : float or array_like
        Geocentric longitude (degrees)
    altitude : float or array_like
        Height (km) above presumed spherical Earth with radius 6371 km.
        
    Returns
    -------
    x, y, z
        numpy arrays of x, y, z locations in km
        
    """

    r = earth_geo_radius + altitude
    x = r * np.cos(np.deg2rad(latitude)) * np.cos(np.deg2rad(longitude))
    y = r * np.cos(np.deg2rad(latitude)) * np.sin(np.deg2rad(longitude))
    z = r * np.sin(np.deg2rad(latitude))

    return x, y, z