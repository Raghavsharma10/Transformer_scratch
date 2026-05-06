def elide_sequence(s, flank=5, elision="..."):
    """trim a sequence to include the left and right flanking sequences of
    size `flank`, with the intervening sequence elided by `elision`.

    >>> elide_sequence("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    'ABCDE...VWXYZ'

    >>> elide_sequence("ABCDEFGHIJKLMNOPQRSTUVWXYZ", flank=3)
    'ABC...XYZ'

    >>> elide_sequence("ABCDEFGHIJKLMNOPQRSTUVWXYZ", elision="..")
    'ABCDE..VWXYZ'

    >>> elide_sequence("ABCDEFGHIJKLMNOPQRSTUVWXYZ", flank=12)
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    >>> elide_sequence("ABCDEFGHIJKLMNOPQRSTUVWXYZ", flank=12, elision=".")
    'ABCDEFGHIJKL.OPQRSTUVWXYZ'

    """

    elided_sequence_len = flank + flank + len(elision)
    if len(s) <= elided_sequence_len:
        return s
    return s[:flank] + elision + s[-flank:]