def encode_string(v, encoding="utf-8"):
    """Returns the given value as a Python byte string (if possible)."""
    if isinstance(encoding, basestring):
        encoding = ((encoding,),) + (("windows-1252",), ("utf-8", "ignore"))
    if isinstance(v, unicode):
        for e in encoding:
            try:
                return v.encode(*e)
            except:
                pass
        return v
    return str(v)