def expand_query(config, kwds):
    """
    Expand `kwds` based on `config.search.query_expander`.

    :type config: .config.Configuration
    :type kwds: dict
    :rtype: dict
    :return: Return `kwds`, modified in place.

    """
    pattern = []
    for query in kwds.pop('pattern', []):
        expansion = config.search.alias.get(query)
        if expansion is None:
            pattern.append(query)
        else:
            parser = SafeArgumentParser()
            search_add_arguments(parser)
            ns = parser.parse_args(expansion)
            for (key, value) in vars(ns).items():
                if isinstance(value, (list, tuple)):
                    if not kwds.get(key):
                        kwds[key] = value
                    else:
                        kwds[key].extend(value)
                else:
                    kwds[key] = value
    kwds['pattern'] = pattern
    return config.search.kwds_adapter(kwds)