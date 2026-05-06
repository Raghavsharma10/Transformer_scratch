def _typelist(x):
    """Helper function converting all items of x to instances."""
    if isinstance(x, collections.Sequence):
        return list(map(_to_instance, x))
    elif isinstance(x, collections.Iterable):
        return x
    return None if x is None else [_to_instance(x)]