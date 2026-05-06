def utf8_bytes_string(s):
    """Convert a string to a bytes string (if necessary, encode in utf8)"""
    if sys.version_info[0] == 2:
        if isinstance(s, str):
            return s
        else:
            return s.encode('utf8')
    else:
        if isinstance(s, str):
            return bytes(s, encoding='utf8')
        else:
            return s