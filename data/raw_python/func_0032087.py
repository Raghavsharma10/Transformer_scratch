def great_distance(**kwargs):
    """
        Named arguments:
        start_latitude  = starting latitude, in DECIMAL DEGREES
        start_longitude = starting longitude, in DECIMAL DEGREES
        end_latitude    = ending latitude, in DECIMAL DEGREES
        end_longitude   = ending longitude, in DECIMAL DEGREES
        rmajor          = radius of earth's major axis. default=6378137.0 (WGS84)
        rminor          = radius of earth's minor axis. default=6356752.3142 (WGS84)

        Returns a dictionaty with:
        'distance' in meters
        'azimuth' in decimal degrees
        'reverse_azimuth' in decimal degrees

    """

    sy     = kwargs.pop('start_latitude')
    sx     = kwargs.pop('start_longitude')
    ey     = kwargs.pop('end_latitude')
    ex     = kwargs.pop('end_longitude')
    rmajor = kwargs.pop('rmajor', 6378137.0)
    rminor = kwargs.pop('rminor', 6356752.3142)
    f      = (rmajor - rminor) / rmajor

    if (np.ma.isMaskedArray(sy) or
        np.ma.isMaskedArray(sx) or
        np.ma.isMaskedArray(ey) or
        np.ma.isMaskedArray(ex)
       ):

        try:
            assert sy.size == sx.size == ey.size == ex.size
        except AttributeError:
            raise ValueError("All or none of the inputs should be masked")
        except AssertionError:
            raise ValueError("When using masked arrays all must be of equal size")

        final_mask = np.logical_not((sy.mask | sx.mask | ey.mask | ex.mask))
        if np.isscalar(final_mask):
            final_mask = np.full(sy.size, final_mask, dtype=bool)
        sy = sy[final_mask]
        sx = sx[final_mask]
        ey = ey[final_mask]
        ex = ex[final_mask]

        if (np.all(sy.mask) or np.all(sx.mask) or np.all(ey.mask) or np.all(ex.mask)) or \
           (sy.size == 0 or sx.size == 0 or ey.size == 0 or ex.size == 0):
            vector_dist = np.vectorize(vinc_dist, otypes=[np.float64])
        else:
            vector_dist = np.vectorize(vinc_dist)

        results = vector_dist(f, rmajor,
                              np.radians(sy),
                              np.radians(sx),
                              np.radians(ey),
                              np.radians(ex))

        d = np.ma.masked_all(final_mask.size, dtype=np.float64)
        a = np.ma.masked_all(final_mask.size, dtype=np.float64)
        ra = np.ma.masked_all(final_mask.size, dtype=np.float64)

        if len(results) == 3:
            d[final_mask] = results[0]
            a[final_mask] = results[1]
            ra[final_mask] = results[2]

    else:
        vector_dist = np.vectorize(vinc_dist)
        d, a, ra = vector_dist(f, rmajor,
                               np.radians(sy),
                               np.radians(sx),
                               np.radians(ey),
                               np.radians(ex))

    return {'distance': d,
            'azimuth': np.degrees(a),
            'reverse_azimuth': np.degrees(ra)}