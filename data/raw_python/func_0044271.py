def tracebacks(score_matrix, traceback_matrix, idx):
    """Calculate the tracebacks for `traceback_matrix` starting at index `idx`.

    Returns: An iterable of tracebacks where each traceback is sequence of
      (index, direction) tuples. Each `index` is an index into
      `traceback_matrix`. `direction` indicates the direction into which the
      traceback "entered" the index.
    """
    return filter(lambda tb: tb != (),
                  _tracebacks(score_matrix,
                              traceback_matrix,
                              idx))