def printable_name(column, path=None):
    """Provided for debug output when rendering conditions.

    User.name[3]["foo"][0]["bar"] -> name[3].foo[0].bar
    """
    pieces = [column.name]
    path = path or path_of(column)
    for segment in path:
        if isinstance(segment, str):
            pieces.append(segment)
        else:
            pieces[-1] += "[{}]".format(segment)
    return ".".join(pieces)