def dumps(obj, big_endian=True):
    """
    Dump a GeoJSON-like `dict` to a WKB string.

    .. note::
        The dimensions of the generated WKB will be inferred from the first
        vertex in the GeoJSON `coordinates`. It will be assumed that all
        vertices are uniform. There are 4 types:

        - 2D (X, Y): 2-dimensional geometry
        - Z (X, Y, Z): 3-dimensional geometry
        - M (X, Y, M): 2-dimensional geometry with a "Measure"
        - ZM (X, Y, Z, M): 3-dimensional geometry with a "Measure"

        If the first vertex contains 2 values, we assume a 2D geometry.
        If the first vertex contains 3 values, this is slightly ambiguous and
        so the most common case is chosen: Z.
        If the first vertex contains 4 values, we assume a ZM geometry.

        The WKT/WKB standards provide a way of differentiating normal (2D), Z,
        M, and ZM geometries (http://en.wikipedia.org/wiki/Well-known_text),
        but the GeoJSON spec does not. Therefore, for the sake of interface
        simplicity, we assume that geometry that looks 3D contains XYZ
        components, instead of XYM.

        If the coordinates list has no coordinate values (this includes nested
        lists, for example, `[[[[],[]], []]]`, the geometry is considered to be
        empty. Geometries, with the exception of points, have a reasonable
        "empty" representation in WKB; however, without knowing the number of
        coordinate values per vertex, the type is ambigious, and thus we don't
        know if the geometry type is 2D, Z, M, or ZM. Therefore in this case
        we expect a `ValueError` to be raised.

    :param dict obj:
        GeoJson-like `dict` object.
    :param bool big_endian:
        Defaults to `True`. If `True`, data values in the generated WKB will
        be represented using big endian byte order. Else, little endian.

    TODO: remove this

    :param str dims:
        Indicates to WKB representation desired from converting the given
        GeoJSON `dict` ``obj``. The accepted values are:

        * '2D': 2-dimensional geometry (X, Y)
        * 'Z': 3-dimensional geometry (X, Y, Z)
        * 'M': 3-dimensional geometry (X, Y, M)
        * 'ZM': 4-dimensional geometry (X, Y, Z, M)

    :returns:
        A WKB binary string representing of the ``obj``.
    """
    geom_type = obj['type']
    meta = obj.get('meta', {})

    exporter = _dumps_registry.get(geom_type)
    if exporter is None:
        _unsupported_geom_type(geom_type)

    # Check for empty geometries. GeometryCollections have a slightly different
    # JSON/dict structure, but that's handled.
    coords_or_geoms = obj.get('coordinates', obj.get('geometries'))
    if len(list(flatten_multi_dim(coords_or_geoms))) == 0:
        raise ValueError(
            'Empty geometries cannot be represented in WKB. Reason: The '
            'dimensionality of the WKB would be ambiguous.'
        )

    return exporter(obj, big_endian, meta)