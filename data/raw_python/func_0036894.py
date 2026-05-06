def load(file,encoding=None):
    """load(file,encoding=None) -> object

    This function reads a tnetstring from a file and parses it into a
    python object.  The file must support the read() method, and this
    function promises not to read more data than necessary.
    """
    #  Read the length prefix one char at a time.
    #  Note that the netstring spec explicitly forbids padding zeros.
    c = file.read(1)
    if not c.isdigit():
        raise ValueError("not a tnetstring: missing or invalid length prefix")
    datalen = ord(c) - ord("0")
    c = file.read(1)
    if datalen != 0:
        while c.isdigit():
            datalen = (10 * datalen) + (ord(c) - ord("0"))
            if datalen > 999999999:
                errmsg = "not a tnetstring: absurdly large length prefix"
                raise ValueError(errmsg)
            c = file.read(1)
    if c != ":":
        raise ValueError("not a tnetstring: missing or invalid length prefix")
    #  Now we can read and parse the payload.
    #  This repeats the dispatch logic of pop() so we can avoid
    #  re-constructing the outermost tnetstring.
    data = file.read(datalen)
    if len(data) != datalen:
        raise ValueError("not a tnetstring: length prefix too big")
    type = file.read(1)
    if type == ",":
        if encoding is not None:
            return data.decode(encoding)
        return data
    if type == "#":
        try:
            return int(data)
        except ValueError:
            raise ValueError("not a tnetstring: invalid integer literal")
    if type == "^":
        try:
            return float(data)
        except ValueError:
            raise ValueError("not a tnetstring: invalid float literal")
    if type == "!":
        if data == "true":
            return True
        elif data == "false":
            return False
        else:
            raise ValueError("not a tnetstring: invalid boolean literal")
    if type == "~":
        if data:
            raise ValueError("not a tnetstring: invalid null literal")
        return None
    if type == "]":
        l = []
        while data:
            (item,data) = pop(data,encoding)
            l.append(item)
        return l
    if type == "}":
        d = {}
        while data:
            (key,data) = pop(data,encoding)
            (val,data) = pop(data,encoding)
            d[key] = val
        return d
    raise ValueError("unknown type tag")