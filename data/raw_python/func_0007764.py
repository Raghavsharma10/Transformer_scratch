def loads(string):
    """
    Construct a GeoJSON `dict` from WKT (`string`).
    """
    sio = StringIO.StringIO(string)
    # NOTE: This is not the intended purpose of `tokenize`, but it works.
    tokens = (x[1] for x in tokenize.generate_tokens(sio.readline))
    tokens = _tokenize_wkt(tokens)
    geom_type_or_srid = next(tokens)
    srid = None
    geom_type = geom_type_or_srid
    if geom_type_or_srid == 'SRID':
        # The geometry WKT contains an SRID header.
        _assert_next_token(tokens, '=')
        srid = int(next(tokens))
        _assert_next_token(tokens, ';')
        # We expected the geometry type to be next:
        geom_type = next(tokens)
    else:
        geom_type = geom_type_or_srid

    importer = _loads_registry.get(geom_type)

    if importer is None:
        _unsupported_geom_type(geom_type)

    peek = six.advance_iterator(tokens)
    if peek == 'EMPTY':
        if geom_type == 'GEOMETRYCOLLECTION':
            return dict(type='GeometryCollection', geometries=[])
        else:
            return dict(type=_type_map_caps_to_mixed[geom_type],
                        coordinates=[])

    # Put the peeked element back on the head of the token generator
    tokens = itertools.chain([peek], tokens)
    result = importer(tokens, string)
    if srid is not None:
        result['meta'] = dict(srid=srid)
    return result