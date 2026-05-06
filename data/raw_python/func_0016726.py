def ordered(obj):
    """
    Return sorted version of nested dicts/lists for comparing.

    Modified from:
    http://stackoverflow.com/a/25851972
    """
    if isinstance(obj, collections.abc.Mapping):
        return sorted((k, ordered(v)) for k, v in obj.items())
    # Special case str since it's a collections.abc.Iterable
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, collections.abc.Iterable):
        return sorted(ordered(x) for x in obj)
    else:
        return obj