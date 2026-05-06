def isInt(num):
    """Returns true if `num` is (sort of) an integer.
    >>> isInt(3) == isInt(3.0) == 1
    True
    >>> isInt(3.2)
    False
    >>> import numpy
    >>> isInt(numpy.array(1))
    True
    >>> isInt(numpy.array([1]))
    False
    """
    try:
        len(num) # FIXME fails for Numeric but Numeric is obsolete
    except:
        try:
            return (num - math.floor(num) == 0) == True
        except: return False
    else: return False