def storeIDToWebID(key, storeid):
    """
    Takes a key (int) and storeid (int) and produces a webid (a 16-character
    str suitable for including in URLs)
    """
    i = key ^ storeid
    l = list('%0.16x' % (i,))
    for nybbleid in range(0, 8):
        a, b = _swapat(key, nybbleid)
        _swap(l, a, b)
    return ''.join(l)