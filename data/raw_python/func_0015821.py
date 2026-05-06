def _escape_char(c):
    "Single char escape. Return the char, escaped if not already legal"
    if isinstance(c, int):
        c = _unichr(c)
    return c if c in LEGAL_CHARS else ESCAPE_FMT % ord(c)