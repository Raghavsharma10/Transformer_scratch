def partition_all(s, sep):
    """
    Uses str.partition() to split every occurrence of sep in s. The returned list does not contain empty strings.
    If sep is a list, all separators are evaluated.

    :param s: The string to split.
    :param sep: A separator string or a list of separator strings.
    :return: A list of parts split by sep
    """
    if isinstance(sep, list):
        parts = _partition_all_internal(s, sep[0])
        sep = sep[1:]

        for s in sep:
            tmp = []
            for p in parts:
                tmp.extend(_partition_all_internal(p, s))
            parts = tmp

        return parts
    else:
        return _partition_all_internal(s, sep)