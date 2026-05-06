def _get_geom_type(type_bytes):
    """Get the GeoJSON geometry type label from a WKB type byte string.

    :param type_bytes:
        4 byte string in big endian byte order containing a WKB type number.
        It may also contain a "has SRID" flag in the high byte (the first type,
        since this is big endian byte order), indicated as 0x20. If the SRID
        flag is not set, the high byte will always be null (0x00).
    :returns:
        3-tuple ofGeoJSON geometry type label, the bytes resprenting the
        geometry type, and a separate "has SRID" flag. If the input
        `type_bytes` contains an SRID flag, it will be removed.

        >>> # Z Point, with SRID flag
        >>> _get_geom_type(b'\\x20\\x00\\x03\\xe9') == (
        ... 'Point', b'\\x00\\x00\\x03\\xe9', True)
        True

        >>> # 2D MultiLineString, without SRID flag
        >>> _get_geom_type(b'\\x00\\x00\\x00\\x05') == (
        ... 'MultiLineString', b'\\x00\\x00\\x00\\x05', False)
        True

    """
    # slice off the high byte, which may contain the SRID flag
    high_byte = type_bytes[0]
    if six.PY3:
        high_byte = bytes([high_byte])
    has_srid = high_byte == b'\x20'
    if has_srid:
        # replace the high byte with a null byte
        type_bytes = as_bin_str(b'\x00' + type_bytes[1:])
    else:
        type_bytes = as_bin_str(type_bytes)

    # look up the geometry type
    geom_type = _BINARY_TO_GEOM_TYPE.get(type_bytes)
    return geom_type, type_bytes, has_srid