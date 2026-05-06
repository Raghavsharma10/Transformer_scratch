def _listify(x):
    """Ensure x is iterable; if not then enclose it in a list and return it."""
    if isinstance(x, string_types):
        return [x]
    elif isinstance(x, collections.Iterable):
        return x
    else:
        return [x]