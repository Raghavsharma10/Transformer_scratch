def lscmp(a, b):
    """ Compares two strings in a cryptographically safe way:
        Runtime is not affected by length of common prefix, so this
        is helpful against timing attacks.

        ..
            from vital.security import lscmp
            lscmp("ringo", "starr")
            # -> False
            lscmp("ringo", "ringo")
            # -> True
        ..
    """
    l = len
    return not sum(0 if x == y else 1 for x, y in zip(a, b)) and l(a) == l(b)