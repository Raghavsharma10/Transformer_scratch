def match_indices(match):
    """Yield index tuples (old_index, new_index) for each place in the match."""
    a, b, size = match
    for i in range(size):
        yield a + i, b + i