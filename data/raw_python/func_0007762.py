def _load_point(big_endian, type_bytes, data_bytes):
    """
    Convert byte data for a Point to a GeoJSON `dict`.

    :param bool big_endian:
        If `True`, interpret the ``data_bytes`` in big endian order, else
        little endian.
    :param str type_bytes:
        4-byte integer (as a binary string) indicating the geometry type
        (Point) and the dimensions (2D, Z, M or ZM). For consistency, these
        bytes are expected to always be in big endian order, regardless of the
        value of ``big_endian``.
    :param str data_bytes:
        Coordinate data in a binary string.

    :returns:
        GeoJSON `dict` representing the Point geometry.
    """
    endian_token = '>' if big_endian else '<'

    if type_bytes == WKB_2D['Point']:
        coords = struct.unpack('%sdd' % endian_token,
                               as_bin_str(take(16, data_bytes)))
    elif type_bytes == WKB_Z['Point']:
        coords = struct.unpack('%sddd' % endian_token,
                               as_bin_str(take(24, data_bytes)))
    elif type_bytes == WKB_M['Point']:
        # NOTE: The use of XYM types geometries is quite rare. In the interest
        # of removing ambiguity, we will treat all XYM geometries as XYZM when
        # generate the GeoJSON. A default Z value of `0.0` will be given in
        # this case.
        coords = list(struct.unpack('%sddd' % endian_token,
                                    as_bin_str(take(24, data_bytes))))
        coords.insert(2, 0.0)
    elif type_bytes == WKB_ZM['Point']:
        coords = struct.unpack('%sdddd' % endian_token,
                               as_bin_str(take(32, data_bytes)))

    return dict(type='Point', coordinates=list(coords))