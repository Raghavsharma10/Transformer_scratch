def haversine(lat_lng1, lat_lng2, native=True):
    if native:
        return _native.haversine(lat_lng1, lat_lng2)

    """Cf https://github.com/mapado/haversine"""
    lat1, lng1 = lat_lng1
    lat2, lng2 = lat_lng2
    lat1, lng1, lat2, lng2 = map(math.radians, (lat1, lng1, lat2, lng2))
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = math.sin(lat * 0.5) ** 2 \
        + math.cos(lat1) * math.cos(lat2) * math.sin(lng * 0.5) ** 2
    return 2 * _AVG_EARTH_RADIUS * math.asin(math.sqrt(d))