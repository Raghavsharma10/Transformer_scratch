def intersection(*args):
    """
    Return the intersection of lists, using the first list to determine item order
    """
    if not args:
        return []

    # remove duplicates from first list whilst preserving order
    base = list(OrderedDict.fromkeys(args[0]))

    if len(args) == 1:
        return base
    else:
        others = set(args[1]).intersection(*args[2:])
        return [e for e in base if e in others]