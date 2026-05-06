def value_for_keypath(obj, path):
    """Get value from walking key path with start object obj.
    """
    val = obj
    for part in path.split('.'):
        match = re.match(list_index_re, part)
        if match is not None:
            val = _extract(val, match.group(1))
            if not isinstance(val, list) and not isinstance(val, tuple):
                raise TypeError('expected list/tuple')
            index = int(match.group(2))
            val = val[index]
        else:
            val = _extract(val, part)
        if val is None:
            return None
    return val