def to_float(s, default=0.0, allow_nan=False):
    """
    Return input converted into a float. If failed, then return ``default``.

    Note that, by default, ``allow_nan=False``, so ``to_float`` will not return
    ``nan``, ``inf``, or ``-inf``.

    Examples::

        >>> to_float('1.5')
        1.5
        >>> to_float(1)
        1.0
        >>> to_float('')
        0.0
        >>> to_float('nan')
        0.0
        >>> to_float('inf')
        0.0
        >>> to_float('-inf', allow_nan=True)
        -inf
        >>> to_float(None)
        0.0
        >>> to_float(0, default='Empty')
        0.0
        >>> to_float(None, default='Empty')
        'Empty'
    """
    try:
        f = float(s)
    except (TypeError, ValueError):
        return default
    if not allow_nan:
        if f != f or f in _infs:
            return default
    return f