def _dump_linestring(obj, decimals):
    """
    Dump a GeoJSON-like LineString object to WKT.

    Input parameters and return value are the LINESTRING equivalent to
    :func:`_dump_point`.
    """
    coords = obj['coordinates']
    ls = 'LINESTRING (%s)'
    ls %= ', '.join(' '.join(_round_and_pad(c, decimals)
                             for c in pt) for pt in coords)
    return ls