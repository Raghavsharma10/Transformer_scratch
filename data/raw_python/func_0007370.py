def iterate_items(dictish):
    """ Return a consistent (key, value) iterable on dict-like objects,
    including lists of tuple pairs.

    Example:

        >>> list(iterate_items({'a': 1}))
        [('a', 1)]
        >>> list(iterate_items([('a', 1), ('b', 2)]))
        [('a', 1), ('b', 2)]
    """
    if hasattr(dictish, 'iteritems'):
        return dictish.iteritems()
    if hasattr(dictish, 'items'):
        return dictish.items()
    return dictish