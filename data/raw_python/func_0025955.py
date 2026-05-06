def ch_handler(offset=0, length=-1, **kw):
    """ Handle standard PRIMARY clipboard access.  Note that offset and length
    are passed as strings.  This differs from CLIPBOARD. """
    global _lastSel

    offset = int(offset)
    length = int(length)
    if length < 0: length = len(_lastSel)
    return _lastSel[offset:offset+length]