def _dump_multipoint(obj, decimals):
    """
    Dump a GeoJSON-like MultiPoint object to WKT.

    Input parameters and return value are the MULTIPOINT equivalent to
    :func:`_dump_point`.
    """
    coords = obj['coordinates']
    mp = 'MULTIPOINT (%s)'
    points = (' '.join(_round_and_pad(c, decimals)
                       for c in pt) for pt in coords)
    # Add parens around each point.
    points = ('(%s)' % pt for pt in points)
    mp %= ', '.join(points)
    return mp