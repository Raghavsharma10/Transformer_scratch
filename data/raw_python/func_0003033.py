def _partition_all_internal(s, sep):
    """
    Uses str.partition() to split every occurrence of sep in s. The returned list does not contain empty strings.

    :param s: The string to split.
    :param sep: A separator string.
    :return: A list of parts split by sep
    """
    parts = list(s.partition(sep))

    # if sep found
    if parts[1] == sep:
        new_parts = partition_all(parts[2], sep)
        parts.pop()
        parts.extend(new_parts)
        return [p for p in parts if p]
    else:
        if parts[0]:
            return [parts[0]]
        else:
            return []