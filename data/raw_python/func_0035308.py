def flatten(nested, containers=(list, tuple)):
    """ Flatten a nested list by yielding its scalar items.
    """
    for item in nested:
        if hasattr(item, "next") or isinstance(item, containers):
            for subitem in flatten(item):
                yield subitem
        else:
            yield item