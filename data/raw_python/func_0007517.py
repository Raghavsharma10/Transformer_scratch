def set_value_for_keypath(obj, path, new_value, preserve_child = False):
    """Set attribute value new_value at key path of start object obj.
    """
    parts = path.split('.')
    last_part = len(parts) - 1
    dst = obj
    for i, part in enumerate(parts):
        match = re.match(list_index_re, part)
        if match is not None:
            dst = _extract(dst, match.group(1))
            if not isinstance(dst, list) and not isinstance(dst, tuple):
                raise TypeError('expected list/tuple')
            index = int(match.group(2))
            if i == last_part:
                dst[index] = new_value
            else:
                dst = dst[index]
        else:
            if i != last_part:
                dst = _extract(dst, part)
            else:
                if isinstance(dst, dict):
                    dst[part] = new_value
                else:
                    if not preserve_child:
                        setattr(dst, part, new_value)
                    else:
                        try:
                            v = getattr(dst, part)
                        except AttributeError:
                            setattr(dst, part, new_value)