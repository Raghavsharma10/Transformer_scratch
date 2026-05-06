def intervals(_min, _max=None, size=1):
    """
    RETURN (min, max) PAIRS OF GIVEN SIZE, WHICH COVER THE _min, _max RANGE
    THE LAST PAIR MAY BE SMALLER
    Yes!  It's just like range(), only cooler!
    """
    if _max == None:
        _max = _min
        _min = 0
    _max = int(mo_math.ceiling(_max))
    _min = int(mo_math.floor(_min))

    output = ((x, min(x + size, _max)) for x in _range(_min, _max, size))
    return output