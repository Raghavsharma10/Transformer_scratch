def decode_string(v, encoding="utf-8"):
    """Returns the given value as a Unicode string (if possible)."""
    if isinstance(encoding, basestring):
        encoding = ((encoding,),) + (("windows-1252",), ("utf-8", "ignore"))
    if isinstance(v, binary_type):
        for e in encoding:
            try:
                return v.decode(*e)
            except:
                pass
        return v
    return unicode(v)