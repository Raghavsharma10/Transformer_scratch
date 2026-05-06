def _str(value, depth):
    """
    FOR DEBUGGING POSSIBLY RECURSIVE STRUCTURES
    """
    output = []
    if depth >0 and _get(value, CLASS) in data_types:
        for k, v in value.items():
            output.append(str(k) + "=" + _str(v, depth - 1))
        return "{" + ",\n".join(output) + "}"
    elif depth >0 and is_list(value):
        for v in value:
            output.append(_str(v, depth-1))
        return "[" + ",\n".join(output) + "]"
    else:
        return str(type(value))