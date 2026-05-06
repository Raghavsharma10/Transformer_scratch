def _splice(value, n):
    """Splice `value` at its center, retaining a total of `n` characters.

    Parameters
    ----------
    value : str
    n : int
        The total length of the returned ends will not be greater than this
        value.  Characters will be dropped from the center to reach this limit.

    Returns
    -------
    A tuple of str: (head, tail).
    """
    if n <= 0:
        raise ValueError("n must be positive")

    value_len = len(value)
    center = value_len // 2
    left, right = value[:center], value[center:]

    if n >= value_len:
        return left, right

    n_todrop = value_len - n
    right_idx = n_todrop // 2
    left_idx = right_idx + n_todrop % 2
    return left[:-left_idx], right[right_idx:]