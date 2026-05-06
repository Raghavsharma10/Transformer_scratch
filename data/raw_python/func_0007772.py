def _dump_multipolygon(obj, decimals):
    """
    Dump a GeoJSON-like MultiPolygon object to WKT.

    Input parameters and return value are the MULTIPOLYGON equivalent to
    :func:`_dump_point`.
    """
    coords = obj['coordinates']
    mp = 'MULTIPOLYGON (%s)'

    polys = (
        # join the polygons in the multipolygon
        ', '.join(
            # join the rings in a polygon,
            # and wrap in parens
            '(%s)' % ', '.join(
                # join the points in a ring,
                # and wrap in parens
                '(%s)' % ', '.join(
                    # join coordinate values of a vertex
                    ' '.join(_round_and_pad(c, decimals) for c in pt)
                    for pt in ring)
                for ring in poly)
            for poly in coords)
    )
    mp %= polys
    return mp