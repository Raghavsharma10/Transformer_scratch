def parFactory(fields, strict=0):

    """parameter factory function

    fields is a list of the comma-separated fields (as in the .par file).
    Each entry is a string or None (indicating that field was omitted.)

    Set the strict parameter to a non-zero value to do stricter parsing
    (to find errors in the input)"""

    if len(fields) < 3 or None in fields[0:3]:
        raise SyntaxError("At least 3 fields must be given")
    type = fields[1]
    if type in _string_types:
        return IrafParS(fields,strict)
    elif type == 'R':
        return StrictParR(fields,1)
    elif type in _real_types:
        return IrafParR(fields,strict)
    elif type == "I":
        return StrictParI(fields,1)
    elif type == "i":
        return IrafParI(fields,strict)
    elif type == "b":
        return IrafParB(fields,strict)
    elif type == "ar":
        return IrafParAR(fields,strict)
    elif type == "ai":
        return IrafParAI(fields,strict)
    elif type == "as":
        return IrafParAS(fields,strict)
    elif type == "ab":
        return IrafParAB(fields,strict)
    elif type[:1] == "a":
        raise SyntaxError("Cannot handle arrays of type %s" % type)
    else:
        raise SyntaxError("Cannot handle parameter type %s" % type)