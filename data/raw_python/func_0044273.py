def _traceback_to_alignment(tb, a, b):
    """Convert a traceback (i.e. as returned by `tracebacks()`) into an alignment
    (i.e. as returned by `align`).

    Arguments:
      tb: A traceback.
      a: the sequence defining the rows in the traceback matrix.
      b: the sequence defining the columns in the traceback matrix.

    Returns: An iterable of (index, index) tupless where ether (but not both)
      tuples can be `None`.
    """
    # We subtract 1 from the indices here because we're translating from the
    # alignment matrix space (which has one extra row and column) to the space
    # of the input sequences.
    for idx, direction in tb:
        if direction == Direction.DIAG:
            yield (idx[0] - 1, idx[1] - 1)
        elif direction == Direction.UP:
            yield (idx[0] - 1, None)
        elif direction == Direction.LEFT:
            yield (None, idx[1] - 1)