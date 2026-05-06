def _dump_multipolygon(obj, big_endian, meta):
    """
    Dump a GeoJSON-like `dict` to a multipolygon WKB string.

    Input parameters and output are similar to :funct:`_dump_point`.
    """
    coords = obj['coordinates']
    vertex = coords[0][0][0]
    num_dims = len(vertex)

    wkb_string, byte_fmt, byte_order = _header_bytefmt_byteorder(
        'MultiPolygon', num_dims, big_endian, meta
    )

    poly_type = _WKB[_INT_TO_DIM_LABEL.get(num_dims)]['Polygon']
    if big_endian:
        poly_type = BIG_ENDIAN + poly_type
    else:
        poly_type = LITTLE_ENDIAN + poly_type[::-1]

    # apped the number of polygons
    wkb_string += struct.pack('%sl' % byte_order, len(coords))

    for polygon in coords:
        # append polygon header
        wkb_string += poly_type
        # append the number of rings in this polygon
        wkb_string += struct.pack('%sl' % byte_order, len(polygon))
        for ring in polygon:
            # append the number of vertices in this ring
            wkb_string += struct.pack('%sl' % byte_order, len(ring))
            for vertex in ring:
                wkb_string += struct.pack(byte_fmt, *vertex)

    return wkb_string