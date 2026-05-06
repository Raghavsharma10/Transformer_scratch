def diff(a, b):
    """
    Performs a longest common substring diff.

    :Parameters:
        a : sequence of `comparable`
            Initial sequence
        b : sequence of `comparable`
            Changed sequence

    :Returns:
        An `iterable` of operations.
    """
    a, b = list(a), list(b)
    opcodes = SM(None, a, b).get_opcodes()
    return parse_opcodes(opcodes)