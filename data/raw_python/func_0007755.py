def loads(string):
    """
    Construct a GeoJSON `dict` from WKB (`string`).

    The resulting GeoJSON `dict` will include the SRID as an integer in the
    `meta` object. This was an arbitrary decision made by `geomet, the
    discussion of which took place here:
    https://github.com/geomet/geomet/issues/28.

    In order to be consistent with other libraries [1] and (deprecated)
    specifications [2], also include the same information in a `crs`
    object. This isn't ideal, but the `crs` member is no longer part of
    the GeoJSON standard, according to RFC7946 [3]. However, it's still
    useful to include this information in GeoJSON payloads because it
    supports conversion to EWKT/EWKB (which are canonical formats used by
    PostGIS and the like).

    Example:

        {'type': 'Point',
         'coordinates': [0.0, 1.0],
         'meta': {'srid': 4326},
         'crs': {'type': 'name', 'properties': {'name': 'EPSG4326'}}}

    NOTE(larsbutler): I'm not sure if it's valid to just prefix EPSG
    (European Petroluem Survey Group) to an SRID like this, but we'll
    stick with it for now until it becomes a problem.

    NOTE(larsbutler): Ideally, we should use URNs instead of this
    notation, according to the new GeoJSON spec [4]. However, in
    order to be consistent with [1], we'll stick with this approach
    for now.

    References:

    [1] - https://github.com/bryanjos/geo/issues/76
    [2] - http://geojson.org/geojson-spec.html#coordinate-reference-system-objects
    [3] - https://tools.ietf.org/html/rfc7946#appendix-B.1
    [4] - https://tools.ietf.org/html/rfc7946#section-4
    """  # noqa
    string = iter(string)
    # endianness = string[0:1]
    endianness = as_bin_str(take(1, string))
    if endianness == BIG_ENDIAN:
        big_endian = True
    elif endianness == LITTLE_ENDIAN:
        big_endian = False
    else:
        raise ValueError("Invalid endian byte: '0x%s'. Expected 0x00 or 0x01"
                         % binascii.hexlify(endianness.encode()).decode())

    endian_token = '>' if big_endian else '<'
    # type_bytes = string[1:5]
    type_bytes = as_bin_str(take(4, string))
    if not big_endian:
        # To identify the type, order the type bytes in big endian:
        type_bytes = type_bytes[::-1]

    geom_type, type_bytes, has_srid = _get_geom_type(type_bytes)
    srid = None
    if has_srid:
        srid_field = as_bin_str(take(4, string))
        [srid] = struct.unpack('%si' % endian_token, srid_field)

    # data_bytes = string[5:]  # FIXME: This won't work for GeometryCollections
    data_bytes = string

    importer = _loads_registry.get(geom_type)

    if importer is None:
        _unsupported_geom_type(geom_type)

    data_bytes = iter(data_bytes)
    result = importer(big_endian, type_bytes, data_bytes)
    if has_srid:
        # As mentioned in the docstring above, include both approaches to
        # indicating the SRID.
        result['meta'] = {'srid': int(srid)}
        result['crs'] = {
            'type': 'name',
            'properties': {'name': 'EPSG%s' % srid},
        }
    return result