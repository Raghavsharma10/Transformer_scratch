def _do_serialize(struct, fmt, encoding):
    """Actually serialize input.

    Args:
        struct: structure to serialize to
        fmt: format to serialize to
        encoding: encoding to use while serializing
    Returns:
        encoded serialized structure
    Raises:
        various sorts of errors raised by libraries while serializing
    """
    res = None
    _check_lib_installed(fmt, 'serialize')

    if fmt == 'ini':
        config = configobj.ConfigObj(encoding=encoding)
        for k, v in struct.items():
            config[k] = v
        res = b'\n'.join(config.write())
    elif fmt in ['json', 'json5']:
        # specify separators to get rid of trailing whitespace
        # specify ensure_ascii to make sure unicode is serialized in \x... sequences,
        #  not in \u sequences
        res = (json if fmt == 'json' else json5).dumps(struct,
                                                       indent=2,
                                                       separators=(',', ': '),
                                                       ensure_ascii=False).encode(encoding)
    elif fmt == 'toml':
        if not _is_utf8(encoding):
            raise AnyMarkupError('toml must always be utf-8 encoded according to specification')
        res = toml.dumps(struct).encode(encoding)
    elif fmt == 'xml':
        # passing encoding argument doesn't encode, just sets the xml property
        res = xmltodict.unparse(struct, pretty=True, encoding='utf-8').encode('utf-8')
    elif fmt == 'yaml':
        res = yaml.safe_dump(struct, encoding='utf-8', default_flow_style=False)
    else:
        raise  # unknown format

    return res