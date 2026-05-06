def webIDToStoreID(key, webid):
    """
    Takes a webid (a 16-character str suitable for including in URLs) and a key
    (an int, a private key for decoding it) and produces a storeID.
    """
    if len(webid) != 16:
        return None
    try:
        int(webid, 16)
    except TypeError:
        return None
    except ValueError:
        return None
    l = list(webid)
    for nybbleid in range(7, -1, -1):
        a, b = _swapat(key, nybbleid)
        _swap(l, b, a)
    i = int(''.join(l), 16)
    return i ^ key