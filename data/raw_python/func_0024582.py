def overlap_status(a, b):
    """Check overlap between two arrays.

    Parameters
    ----------
    a, b : array-like
        Arrays to check. Assumed to be in the same unit.

    Returns
    -------
    result : {'full', 'partial', 'none'}
        * 'full' - ``a`` is within or same as ``b``
        * 'partial' - ``a`` partially overlaps with ``b``
        * 'none' - ``a`` does not overlap ``b``

    """
    # Get the endpoints
    a1, a2 = a.min(), a.max()
    b1, b2 = b.min(), b.max()

    # Do the comparison
    if a1 >= b1 and a2 <= b2:
        result = 'full'
    elif a2 < b1 or b2 < a1:
        result = 'none'
    else:
        result = 'partial'

    return result