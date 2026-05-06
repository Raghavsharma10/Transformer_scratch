def mp_serialize_dict(
        bundle: dict,
        separator: str = '.',
        serialize: t.Optional[t.Callable] = dump_yaml,
        value_prefix: str = '::YAML::\n') -> t.List[t.Tuple[str, bytes]]:
    """
    Transforms a given ``bundle`` into a *sorted* list of tuples with materialized value paths and values:
    ``('path.to.value', b'<some>')``. If the ``<some>`` value is not an instance of a basic type, it's serialized
    with ``serialize`` callback. If this value is an empty string, it's serialized anyway to enforce correct
    type if storage backend does not support saving empty strings.

    :param bundle: a dict to materialize
    :param separator: build paths with a given separator
    :param serialize: a method to serialize non-basic types, default is ``yaml.dump``
    :param value_prefix: a prefix for non-basic serialized types
    :return: a list of tuples ``(mat_path, b'value')``

    ::

        sample = {
            'bool_flag': '',  # flag
            'unicode': 'вася',
            'none_value': None,
            'debug': True,
            'mixed': ['ascii', 'юникод', 1, {'d': 1}, {'b': 2}],
            'nested': {
                'a': {
                    'b': 2,
                    'c': b'bytes',
                }
            }
        }

        result = mp_serialize_dict(sample, separator='/')
        assert result == [
            ('nested/a/b', b'2'),
            ('nested/a/c', b'bytes'),
            ('bool_flag', b"::YAML::\\n''\\n"),
            ('debug', b'true'),
            ('mixed', b'::YAML::\\n- ascii\\n- '
                      b'"\\\\u044E\\\\u043D\\\\u0438\\\\u043A\\\\u043E\\\\u0434"\\n- 1\\n- '
                      b'{d: 1}\\n- {b: 2}\\n'),
            ('none_value', None),
            ('unicode', b'\\xd0\\xb2\\xd0\\xb0\\xd1\\x81\\xd1\\x8f')
        ]
    """

    md = materialize_dict(bundle, separator=separator)
    res = []
    for path, value in md:
        # have to serialize values (value should be None or a string / binary data)
        if value is None:
            pass
        elif isinstance(value, str) and value != '':
            # check for value != '' used to armor empty string with forced serialization
            # since it can be not recognized by a storage backend
            pass
        elif isinstance(value, bytes):
            pass
        elif isinstance(value, bool):
            value = str(value).lower()
        elif isinstance(value, (int, float, Decimal)):
            value = str(value)
        else:
            value = (value_prefix + serialize(value))

        if isinstance(value, str):
            value = value.encode()

        res.append((path, value))

    return res