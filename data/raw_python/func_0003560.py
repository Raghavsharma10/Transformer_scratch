def sum_all(iterable, start):
    """Sum up an iterable starting with a start value.

    In contrast to :func:`sum`, this also works on other types like
    :class:`lists <list>` and :class:`sets <set>`.
    """
    if hasattr(start, "__add__"):
        for value in iterable:
            start += value
    else:
        for value in iterable:
            start |= value
    return start