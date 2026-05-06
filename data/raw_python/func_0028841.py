def item_str_path_or(default, keys, dct):
    """
        Given a string of path segments separated by ., splits them into an array. Int strings are converted
        to numbers to serve as an array index
    :param default: Value if any part yields None or undefined
    :param keys: e.g. 'foo.bar.1.goo'
    :param dct: e.g. dict(foo=dict(bar=[dict(goo='a'), dict(goo='b')])
    :return: The resolved value or an error. E.g. for above the result would be b
    """
    return item_path_or(default, map(lambda segment: int(segment) if isint(segment) else segment, keys.split('.')), dct)