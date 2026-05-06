def floatSlist(value):
    """ Converts float to signed list. """
    slist = ['+', 0, 0, 0, 0]
    if value < 0:
        slist[0] = '-'
    value = abs(value)
    for i in range(1,5):
        slist[i] = math.floor(value)
        value = (value - slist[i]) * 60
    return _roundSlist(slist)