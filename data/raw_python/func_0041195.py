def get_alldictkeys(ddict, parent=None):
    """
    Get all keys in a dict
    """
    parent = [] if parent is None else parent

    if not isinstance(ddict, dict):
        return [tuple(parent)]
    return reduce(
        list.__add__,
        [get_alldictkeys(v, parent + [k]) for k, v in ddict.items()],
        [])