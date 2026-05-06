def is_traditional(s):
    """Check if a string's Chinese characters are Traditional.

    This is equivalent to:
        >>> identify('foo') in (TRADITIONAL, BOTH)

    """
    chinese = _get_hanzi(s)
    if not chinese:
        return False
    elif chinese.issubset(_SHARED_CHARACTERS):
        return True
    elif chinese.issubset(_TRADITIONAL_CHARACTERS):
        return True
    return False