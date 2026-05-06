def dict_to_object(d):
    """Recursively converts a dict to an object"""
    top = type('CreateSendModel', (object,), d)
    seqs = tuple, list, set, frozenset
    for i, j in d.items():
        if isinstance(j, dict):
            setattr(top, i, dict_to_object(j))
        elif isinstance(j, seqs):
            setattr(top, i, type(j)(dict_to_object(sj) if isinstance(sj, dict) else sj for sj in j))
        else:
            setattr(top, i, j)
    return top