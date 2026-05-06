def unwraplist(v):
    """
    LISTS WITH ZERO AND ONE element MAP TO None AND element RESPECTIVELY
    """
    if is_list(v):
        if len(v) == 0:
            return None
        elif len(v) == 1:
            return unwrap(v[0])
        else:
            return unwrap(v)
    else:
        return unwrap(v)