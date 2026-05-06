def positionIf(pred, seq):
    """
    >>> positionIf(lambda x: x > 3, range(10))
    4
    """
    for i,e in enumerate(seq):
        if pred(e):
            return i
    return -1