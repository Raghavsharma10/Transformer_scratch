def split(s):
    """
    Split a string into a list, respecting any quoted strings inside
    Uses ``shelx.split`` which has a bad habbit of inserting null bytes where they are not wanted
    """
    return map(lambda w: filter(lambda c: c != '\x00', w), lexsplit(s))