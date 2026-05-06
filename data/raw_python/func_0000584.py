def compare_schemas(one, two):
    """Compare two structures that represents JSON schemas.

    For comparison you can't use normal comparison, because in JSON schema
    lists DO NOT keep order (and Python lists do), so this must be taken into
    account during comparison.

    Note this wont check all configurations, only first one that seems to
    match, which can lead to wrong results.

    :param one: First schema to compare.
    :param two: Second schema to compare.
    :rtype: `bool`

    """
    one = _normalize_string_type(one)
    two = _normalize_string_type(two)

    _assert_same_types(one, two)

    if isinstance(one, list):
        return _compare_lists(one, two)
    elif isinstance(one, dict):
        return _compare_dicts(one, two)
    elif isinstance(one, SCALAR_TYPES):
        return one == two
    elif one is None:
        return one is two
    else:
        raise RuntimeError('Not allowed type "{type}"'.format(
            type=type(one).__name__))