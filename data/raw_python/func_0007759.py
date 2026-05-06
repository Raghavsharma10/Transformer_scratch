def _dump_multipoint(obj, big_endian, meta):
    """
    Dump a GeoJSON-like `dict` to a multipoint WKB string.

    Input parameters and output are similar to :funct:`_dump_point`.
    """
    coords = obj['coordinates']
    vertex = coords[0]
    num_dims = len(vertex)

    wkb_string, byte_fmt, byte_order = _header_bytefmt_byteorder(
        'MultiPoint', num_dims, big_endian, meta
    )

    point_type = _WKB[_INT_TO_DIM_LABEL.get(num_dims)]['Point']
    if big_endian:
        point_type = BIG_ENDIAN + point_type
    else:
        point_type = LITTLE_ENDIAN + point_type[::-1]

    wkb_string += struct.pack('%sl' % byte_order, len(coords))
    for vertex in coords:
        # POINT type strings
        wkb_string += point_type
        wkb_string += struct.pack(byte_fmt, *vertex)

    return wkb_string