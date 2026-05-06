def is_simplified(s):
    """Check if a string's Chinese characters are Simplified.

    This is equivalent to:
        >>> identify('foo') in (SIMPLIFIED, BOTH)

    """
    chinese = _get_hanzi(s)
    if not chinese:
        return False
    elif chinese.issubset(_SHARED_CHARACTERS):
        return True
    elif chinese.issubset(_SIMPLIFIED_CHARACTERS):
        return True
    return False