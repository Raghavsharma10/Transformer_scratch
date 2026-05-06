def parse(s):
    r"""
    Returns a list of strings or format dictionaries to describe the strings.

    May raise a ValueError if it can't be parsed.

    >>> parse(">>> []")
    ['>>> []']
    >>> #parse("\x1b[33m[\x1b[39m\x1b[33m]\x1b[39m\x1b[33m[\x1b[39m\x1b[33m]\x1b[39m\x1b[33m[\x1b[39m\x1b[33m]\x1b[39m\x1b[33m[\x1b[39m")
    """
    stuff = []
    rest = s
    while True:
        front, token, rest = peel_off_esc_code(rest)
        if front:
            stuff.append(front)
        if token:
            try:
                tok = token_type(token)
                if tok:
                    stuff.extend(tok)
            except ValueError:
                raise ValueError("Can't parse escape sequence: %r %r %r %r" % (s, repr(front), token, repr(rest)))
        if not rest:
            break
    return stuff