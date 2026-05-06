def set_default(*params):
    """
    INPUT dicts IN PRIORITY ORDER
    UPDATES FIRST dict WITH THE MERGE RESULT, WHERE MERGE RESULT IS DEFINED AS:
    FOR EACH LEAF, RETURN THE HIGHEST PRIORITY LEAF VALUE
    """
    p0 = params[0]
    agg = p0 if p0 or _get(p0, CLASS) in data_types else {}
    for p in params[1:]:
        p = unwrap(p)
        if p is None:
            continue
        _all_default(agg, p, seen={})
    return wrap(agg)