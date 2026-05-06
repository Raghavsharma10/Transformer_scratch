def great_circle(**kwargs):
    """
        Named arguments:
        distance  = distance to travel, or numpy array of distances
        azimuth   = angle, in DEGREES of HEADING from NORTH, or numpy array of azimuths
        latitude  = latitude, in DECIMAL DEGREES, or numpy array of latitudes
        longitude = longitude, in DECIMAL DEGREES, or numpy array of longitudes
        rmajor    = radius of earth's major axis. default=6378137.0 (WGS84)
        rminor    = radius of earth's minor axis. default=6356752.3142 (WGS84)

        Returns a dictionary with:
        'latitude' in decimal degrees
        'longitude' in decimal degrees
        'reverse_azimuth' in decimal degrees

    """

    distance  = kwargs.pop('distance')
    azimuth   = np.radians(kwargs.pop('azimuth'))
    latitude  = np.radians(kwargs.pop('latitude'))
    longitude = np.radians(kwargs.pop('longitude'))
    rmajor    = kwargs.pop('rmajor', 6378137.0)
    rminor    = kwargs.pop('rminor', 6356752.3142)
    f         = (rmajor - rminor) / rmajor

    vector_pt = np.vectorize(vinc_pt)
    lat_result, lon_result, angle_result = vector_pt(f, rmajor,
                                                     latitude,
                                                     longitude,
                                                     azimuth,
                                                     distance)
    return {'latitude': np.degrees(lat_result),
            'longitude': np.degrees(lon_result),
            'reverse_azimuth': np.degrees(angle_result)}