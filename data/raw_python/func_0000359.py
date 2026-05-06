def _list_dict(l: Iterator[str], case_insensitive: bool = False):
    """
    return a dictionary with all items of l being the keys of the dictionary

    If argument case_insensitive is non-zero ldap.cidict.cidict will be
    used for case-insensitive string keys
    """
    if case_insensitive:
        raise NotImplementedError()
        d = tldap.dict.CaseInsensitiveDict()
    else:
        d = {}
    for i in l:
        d[i] = None
    return d