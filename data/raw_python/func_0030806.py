def parseJuiceHeaders(lines):
    """
    Create a JuiceBox from a list of header lines.

    @param lines: a list of lines.
    """
    b = JuiceBox()
    bodylen = 0
    key = None
    for L in lines:
        if L[0] == ' ':
            # continuation
            assert key is not None
            b[key] += '\r\n'+L[1:]
            continue
        parts = L.split(': ', 1)
        if len(parts) != 2:
            raise MalformedJuiceBox("Wrong number of parts: %r" % (L,))
        key, value = parts
        key = normalizeKey(key)
        b[key] = value
    return int(b.pop(LENGTH, 0)), b