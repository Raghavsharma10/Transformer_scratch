def _product(k, v):
    """
        Perform the product between two objects
        even if they don't support iteration
    """
    if not _can_iterate(k):
        k = [k]
    if not _can_iterate(v):
        v = [v]
    return list(product(k, v))