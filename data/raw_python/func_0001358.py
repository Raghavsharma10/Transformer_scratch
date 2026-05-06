def dotkey(obj: dict, path: str, default=None, separator='.'):
    """
    Provides an interface to traverse nested dict values by dot-separated paths. Wrapper for ``dpath.util.get``.

    :param obj: dict like ``{'some': {'value': 3}}``
    :param path: ``'some.value'``
    :param separator: ``'.'`` or ``'/'`` or whatever
    :param default: default for KeyError
    :return: dict value or default value
    """
    try:
        return get(obj, path, separator=separator)
    except KeyError:
        return default