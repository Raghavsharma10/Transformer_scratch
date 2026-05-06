def referrers(igt, id, refattrs=None):
    """
    Return a list of ids denoting objects (tiers or items) in `igt` that
    refer to the given `id`. In other words, if 'b1' refers to 'a1',
    then `referrers(igt, 'a1')` returns `['b1']`.
    """
    if refattrs is None:
        result = {}
    else:
        result = {ra: [] for ra in refattrs}

    # if the id is a tier, only look at tiers; otherwise only look at items
    try:
        obj = igt[id]
        others = igt.tiers
    except KeyError:
        obj = igt.get_item(id)
        others = [i for t in igt.tiers for i in t.items]

    if obj is None:
        raise XigtLookupError(id)

    for other in others:
        if other.id is None:
            continue  # raise a warning?

        _refattrs = refattrs
        if _refattrs is None:
            _refattrs = other.allowed_reference_attributes()

        attrget = other.attributes.get  # just loop optimization
        for ra in _refattrs:
            result.setdefault(ra, [])
            if id in ids(attrget(ra, '')):
                result[ra].append(other.id)
    return result