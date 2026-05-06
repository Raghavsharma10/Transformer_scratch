def get_nonmatching_blocks(matching_blocks):
    """Given a list of matching blocks, output the gaps between them.

    Non-matches have the format (alo, ahi, blo, bhi). This specifies two index
    ranges, one in the A sequence, and one in the B sequence.
    """
    i = j = 0
    for match in matching_blocks:
        a, b, size = match
        yield (i, a, j, b)
        i = a + size
        j = b + size