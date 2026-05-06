def _stdin_(p):
    """Takes input from user. Works for Python 2 and 3."""
    _v = sys.version[0]
    return input(p) if _v is '3' else raw_input(p)