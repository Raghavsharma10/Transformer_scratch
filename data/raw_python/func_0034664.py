def _sortObjects(orderby='created', **kwargs):
    """Sorts lists of objects and combines them into a single list"""
    o = []
    
    for m in kwargs.values():
        for l in iter(m):
            o.append(l)
    o = list(set(o))
    sortfunc = _sortByCreated if orderby == 'created' else _sortByModified
    if six.PY2:
        o.sort(sortfunc)
    else:
        o.sort(key=functools.cmp_to_key(sortfunc))

    return o