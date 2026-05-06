def serialize_dict_keys(d, prefix=""):
    """returns all the keys in a dictionary.

    >>> serialize_dict_keys({"a": {"b": {"c": 1, "b": 2} } })
    ['a', 'a.b', 'a.b.c', 'a.b.b']
    """
    keys = []
    for k, v in d.iteritems():
        fqk = '%s%s' % (prefix, k)
        keys.append(fqk)
        if isinstance(v, dict):
            keys.extend(serialize_dict_keys(v, prefix="%s." % fqk))

    return keys