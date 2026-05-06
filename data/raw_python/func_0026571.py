def Distance(lat1, lon1, lat2, lon2):
    """Get distance between pairs of lat-lon points"""

    az12, az21, dist = wgs84_geod.inv(lon1, lat1, lon2, lat2)
    return az21, dist