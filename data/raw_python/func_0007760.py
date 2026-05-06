def _dump_multilinestring(obj, big_endian, meta):
    """
    Dump a GeoJSON-like `dict` to a multilinestring WKB string.

    Input parameters and output are similar to :funct:`_dump_point`.
    """
    coords = obj['coordinates']
    vertex = coords[0][0]
    num_dims = len(vertex)

    wkb_string, byte_fmt, byte_order = _header_bytefmt_byteorder(
        'MultiLineString', num_dims, big_endian, meta
    )

    ls_type = _WKB[_INT_TO_DIM_LABEL.get(num_dims)]['LineString']
    if big_endian:
        ls_type = BIG_ENDIAN + ls_type
    else:
        ls_type = LITTLE_ENDIAN + ls_type[::-1]

    # append the number of linestrings
    wkb_string += struct.pack('%sl' % byte_order, len(coords))

    for linestring in coords:
        wkb_string += ls_type
        # append the number of vertices in each linestring
        wkb_string += struct.pack('%sl' % byte_order, len(linestring))
        for vertex in linestring:
            wkb_string += struct.pack(byte_fmt, *vertex)

    return wkb_string