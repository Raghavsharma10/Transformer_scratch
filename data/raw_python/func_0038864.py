def union(*args):
    """
    Return the union of lists, ordering by first seen in any list
    """
    if not args:
        return []

    base = args[0]
    for other in args[1:]:
        base.extend(other)

    return list(OrderedDict.fromkeys(base))