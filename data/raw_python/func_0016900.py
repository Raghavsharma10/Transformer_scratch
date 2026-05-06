def sources_to_nr_vars(sources):
    """
    Converts a source type to number of sources mapping into
    a source numbering variable to number of sources mapping.

    If, for example, we have 'point', 'gaussian' and 'sersic'
    source types, then passing the following dict as an argument

    sources_to_nr_vars({'point':10, 'gaussian': 20})

    will return an OrderedDict

    {'npsrc': 10, 'ngsrc': 20, 'nssrc': 0 }
    """

    sources = default_sources(**sources)

    try:
        return OrderedDict((SOURCE_VAR_TYPES[name], nr)
            for name, nr in sources.iteritems())
    except KeyError as e:
        raise KeyError((
            'No source type ''%s'' is '
            'registered. Valid source types '
            'are %s') % (e, SOURCE_VAR_TYPES.keys()))