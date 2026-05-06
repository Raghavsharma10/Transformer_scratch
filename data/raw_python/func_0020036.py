def to_bytes(s, encoding=None, errors='strict'):
    """Returns a bytestring version of 's',
encoded as specified in 'encoding'."""
    encoding = encoding or 'utf-8'
    if isinstance(s, bytes):
        if encoding != 'utf-8':
            return s.decode('utf-8', errors).encode(encoding, errors)
        else:
            return s
    if not is_string(s):
        s = string_type(s)
    return s.encode(encoding, errors)