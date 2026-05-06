def pop(string,encoding=None):
    """pop(string,encoding=None) -> (object, remain)

    This function parses a tnetstring into a python object.
    It returns a tuple giving the parsed object and a string
    containing any unparsed data from the end of the string.
    """
    #  Parse out data length, type and remaining string.
    try:
        (dlen,rest) = string.split(":",1)
        dlen = int(dlen)
    except ValueError:
        raise ValueError("not a tnetstring: missing or invalid length prefix")
    try:
        (data,type,remain) = (rest[:dlen],rest[dlen],rest[dlen+1:])
    except IndexError:
        #  This fires if len(rest) < dlen, meaning we don't need
        #  to further validate that data is the right length.
        raise ValueError("not a tnetstring: invalid length prefix")
    #  Parse the data based on the type tag.
    if type == ",":
        if encoding is not None:
            return (data.decode(encoding),remain)
        return (data,remain)
    if type == "#":
        try:
            return (int(data),remain)
        except ValueError:
            raise ValueError("not a tnetstring: invalid integer literal")
    if type == "^":
        try:
            return (float(data),remain)
        except ValueError:
            raise ValueError("not a tnetstring: invalid float literal")
    if type == "!":
        if data == "true":
            return (True,remain)
        elif data == "false":
            return (False,remain)
        else:
            raise ValueError("not a tnetstring: invalid boolean literal")
    if type == "~":
        if data:
            raise ValueError("not a tnetstring: invalid null literal")
        return (None,remain)
    if type == "]":
        l = []
        while data:
            (item,data) = pop(data,encoding)
            l.append(item)
        return (l,remain)
    if type == "}":
        d = {}
        while data:
            (key,data) = pop(data,encoding)
            (val,data) = pop(data,encoding)
            d[key] = val
        return (d,remain)
    raise ValueError("unknown type tag")