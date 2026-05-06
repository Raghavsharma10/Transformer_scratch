def unique(iterable):
    """ Returns a list copy in which each item occurs only once (in-order).
    """
    seen = set()
    return [x for x in iterable if x not in seen and not seen.add(x)]