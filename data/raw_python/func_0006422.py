def iterable(item):
    """generate iterable from item, but leaves out strings

    """
    if isinstance(item, collections.Iterable) and not isinstance(item, basestring):
        return item
    else:
        return [item]