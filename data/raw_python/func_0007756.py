def _header_bytefmt_byteorder(geom_type, num_dims, big_endian, meta=None):
    """
    Utility function to get the WKB header (endian byte + type header), byte
    format string, and byte order string.
    """
    dim = _INT_TO_DIM_LABEL.get(num_dims)
    if dim is None:
        pass  # TODO: raise

    type_byte_str = _WKB[dim][geom_type]
    srid = meta.get('srid')
    if srid is not None:
        # Add the srid flag
        type_byte_str = SRID_FLAG + type_byte_str[1:]

    if big_endian:
        header = BIG_ENDIAN
        byte_fmt = b'>'
        byte_order = '>'
    else:
        header = LITTLE_ENDIAN
        byte_fmt = b'<'
        byte_order = '<'
        # reverse the byte ordering for little endian
        type_byte_str = type_byte_str[::-1]

    header += type_byte_str
    if srid is not None:
        srid = int(srid)

        if big_endian:
            srid_header = struct.pack('>i', srid)
        else:
            srid_header = struct.pack('<i', srid)
        header += srid_header
    byte_fmt += b'd' * num_dims

    return header, byte_fmt, byte_order