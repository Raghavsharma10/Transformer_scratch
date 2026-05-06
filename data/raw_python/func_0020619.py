def exclude_from(l, containing = [], equal_to = []):
    """Exclude elements in list l containing any elements from list ex.
    Example:
        >>> l = ['bob', 'r', 'rob\r', '\r\nrobert']
        >>> containing = ['\n', '\r']
        >>> equal_to = ['r']
        >>> exclude_from(l, containing, equal_to)
        ['bob']
    """
      
    cont = lambda li: any(c in li for c in containing)
    eq = lambda li: any(e == li for e in equal_to)
    return [li for li in l if not (cont(li) or eq(li))]