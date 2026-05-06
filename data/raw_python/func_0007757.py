def _dump_point(obj, big_endian, meta):
    """
    Dump a GeoJSON-like `dict` to a point WKB string.

    :param dict obj:
        GeoJson-like `dict` object.
    :param bool big_endian:
        If `True`, data values in the generated WKB will be represented using
        big endian byte order. Else, little endian.
    :param dict meta:
        Metadata associated with the GeoJSON object. Currently supported
        metadata:

        - srid: Used to support EWKT/EWKB. For example, ``meta`` equal to
          ``{'srid': '4326'}`` indicates that the geometry is defined using
          Extended WKT/WKB and that it bears a Spatial Reference System
          Identifier of 4326. This ID will be encoded into the resulting
          binary.

        Any other meta data objects will simply be ignored by this function.

    :returns:
        A WKB binary string representing of the Point ``obj``.
    """
    coords = obj['coordinates']
    num_dims = len(coords)

    wkb_string, byte_fmt, _ = _header_bytefmt_byteorder(
        'Point', num_dims, big_endian, meta
    )

    wkb_string += struct.pack(byte_fmt, *coords)
    return wkb_string