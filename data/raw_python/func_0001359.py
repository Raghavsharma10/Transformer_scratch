def _materialize_dict(bundle: dict, separator: str = '.') -> t.Generator[t.Tuple[str, t.Any], None, None]:
    """
    Traverses and transforms a given dict ``bundle`` into tuples of ``(key_path, value)``.

    :param bundle: a dict to traverse
    :param separator: build paths with a given separator
    :return: a generator of tuples ``(materialized_path, value)``

    Example:
    >>> list(_materialize_dict({'test': {'path': 1}, 'key': 'val'}, '.'))
    >>> [('key', 'val'), ('test.path', 1)]
    """
    for path_prefix, v in bundle.items():
        if not isinstance(v, dict):
            yield str(path_prefix), v
            continue

        for nested_path, nested_val in _materialize_dict(v, separator=separator):
            yield '{0}{1}{2}'.format(path_prefix, separator, nested_path), nested_val