def unescape(value):
    """
    Inverse of escape.
    """
    pattern = ESCAPE_FMT.replace('%02X', '(?P<code>[0-9A-Fa-f]{2})')
    # the pattern must be bytes to operate on bytes
    pattern_bytes = pattern.encode('ascii')
    re_esc = re.compile(pattern_bytes)
    return re_esc.sub(_unescape_code, value.encode('ascii')).decode('utf-8')