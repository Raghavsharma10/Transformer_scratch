def default_sources(**kwargs):
    """
    Returns a dictionary mapping source types
    to number of sources. If the number of sources
    for the source type is supplied in the kwargs
    these will be placed in the dictionary.

    e.g. if we have 'point', 'gaussian' and 'sersic'
    source types, then

    default_sources(point=10, gaussian=20)

    will return an OrderedDict {'point': 10, 'gaussian': 20, 'sersic': 0}
    """
    S = OrderedDict()
    total = 0

    invalid_types = [t for t in kwargs.keys() if t not in SOURCE_VAR_TYPES]

    for t in invalid_types:
        montblanc.log.warning('Source type %s is not yet '
            'implemented in montblanc. '
            'Valid source types are %s' % (t, SOURCE_VAR_TYPES.keys()))

    # Zero all source types
    for k, v in SOURCE_VAR_TYPES.iteritems():
        # Try get the number of sources for this source
        # from the kwargs
        value = kwargs.get(k, 0)

        try:
            value = int(value)
        except ValueError:
            raise TypeError(('Supplied value %s '
                'for source %s cannot be '
                'converted to an integer') % \
                    (value, k))

        total += value
        S[k] = value

    # Add a point source if no others exist
    if total == 0:
        S[POINT_TYPE] = 1

    return S