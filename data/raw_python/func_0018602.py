def merge_blocks(a_blocks, b_blocks):
    """Given two lists of blocks, combine them, in the proper order.

    Ensure that there are no overlaps, and that they are for sequences of the
    same length.
    """
    # Check sentinels for sequence length.
    assert a_blocks[-1][2] == b_blocks[-1][2] == 0 # sentinel size is 0
    assert a_blocks[-1] == b_blocks[-1]
    combined_blocks = sorted(list(set(a_blocks + b_blocks)))
    # Check for overlaps.
    i = j = 0
    for a, b, size in combined_blocks:
        assert i <= a
        assert j <= b
        i = a + size
        j = b + size
    return combined_blocks