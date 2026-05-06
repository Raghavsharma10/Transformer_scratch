def _to_unicode_scalar_value(s):
    """
    Helper function for converting a character or surrogate pair into a Unicode scalar value e.g.
    "\ud800\udc00" -> 0x10000

    The algorithm can be found in older versions of the Unicode Standard.

    https://unicode.org/versions/Unicode3.0.0/ch03.pdf, Section 3.7, D28

    Unicode scalar value: a number N from 0 to 0x10FFFF is defined by applying the following algorithm to a
    character sequence S:
    If S is a single, non-surrogate value U:
    N = U
    If S is a surrogate pair H, L:
    N = (H - 0xD800) * 0x0400 + (L - 0xDC00) + 0x10000

    :param s:
    :return:
    """
    if len(s) == 1:
        return ord(s)
    elif len(s) == 2:
        return (ord(s[0]) - 0xD800) * 0x0400 + (ord(s[1]) - 0xDC00) + 0x10000
    else:
        raise ValueError