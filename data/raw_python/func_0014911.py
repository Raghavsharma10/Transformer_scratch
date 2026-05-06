def unique(iterable):
    r"""Returns all unique items in `iterable` in the *same* order (only works
    if items in `seq` are hashable).
    """
    d = {}
    return (d.setdefault(x,x) for x in iterable if x not in d)