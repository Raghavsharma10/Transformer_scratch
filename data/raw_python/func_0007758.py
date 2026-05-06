def _dump_linestring(obj, big_endian, meta):
    """
    Dump a GeoJSON-like `dict` to a linestring WKB string.

    Input parameters and output are similar to :func:`_dump_point`.
    """
    coords = obj['coordinates']
    vertex = coords[0]
    # Infer the number of dimensions from the first vertex
    num_dims = len(vertex)

    wkb_string, byte_fmt, byte_order = _header_bytefmt_byteorder(
        'LineString', num_dims, big_endian, meta
    )
    # append number of vertices in linestring
    wkb_string += struct.pack('%sl' % byte_order, len(coords))

    for vertex in coords:
        wkb_string += struct.pack(byte_fmt, *vertex)

    return wkb_string