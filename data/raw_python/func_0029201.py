def flatten(lis):
    """Given a list, possibly nested to any level, return it flattened."""
    new_lis = []
    for item in lis:
        if isinstance(item, collections.Sequence) and not isinstance(item, basestring):
            new_lis.extend(flatten(item))
        else:
            new_lis.append(item)
    return new_lis