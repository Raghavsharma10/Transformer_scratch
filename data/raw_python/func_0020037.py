def to_string(s, encoding=None, errors='strict'):
    """Inverse of to_bytes"""
    encoding = encoding or 'utf-8'
    if isinstance(s, bytes):
        return s.decode(encoding, errors)
    if not is_string(s):
        s = string_type(s)
    return s