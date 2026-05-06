def truncate(n):
    """
    Removes trailing zeros.

    Args:
        n:  The number to truncate.
            This number should be in the following form:
            (..., '.', int, int, int, ..., 0)
    Returns:
        n with all trailing zeros removed

    >>> truncate((9, 9, 9, '.', 9, 9, 9, 9, 0, 0, 0, 0))
    (9, 9, 9, '.', 9, 9, 9, 9)
    >>> truncate(('.',))
    ('.',)
    """
    count = 0
    for digit in n[-1::-1]:
        if digit != 0:
            break
        count += 1
    return n[:-count] if count > 0 else n