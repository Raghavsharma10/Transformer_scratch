def str2dict(dotted_str, value=None, separator='.'):
    """ Convert dotted string to dict splitting by :separator: """
    dict_ = {}
    parts = dotted_str.split(separator)
    d, prev = dict_, None
    for part in parts:
        prev = d
        d = d.setdefault(part, {})
    else:
        if value is not None:
            prev[part] = value
    return dict_