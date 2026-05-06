def _dump_polygon(obj, decimals):
    """
    Dump a GeoJSON-like Polygon object to WKT.

    Input parameters and return value are the POLYGON equivalent to
    :func:`_dump_point`.
    """
    coords = obj['coordinates']
    poly = 'POLYGON (%s)'
    rings = (', '.join(' '.join(_round_and_pad(c, decimals)
                                for c in pt) for pt in ring)
             for ring in coords)
    rings = ('(%s)' % r for r in rings)
    poly %= ', '.join(rings)
    return poly