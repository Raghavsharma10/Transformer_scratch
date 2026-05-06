def _dump_point(obj, decimals):
    """
    Dump a GeoJSON-like Point object to WKT.

    :param dict obj:
        A GeoJSON-like `dict` representing a Point.
    :param int decimals:
        int which indicates the number of digits to display after the
        decimal point when formatting coordinates.

    :returns:
        WKT representation of the input GeoJSON Point ``obj``.
    """
    coords = obj['coordinates']
    pt = 'POINT (%s)' % ' '.join(_round_and_pad(c, decimals)
                                 for c in coords)
    return pt