def to_d(l):
    """
    Converts list of dicts to dict.
    """
    _d = {}
    for x in l:
        for k, v in x.items():
            _d[k] = v
    return _d