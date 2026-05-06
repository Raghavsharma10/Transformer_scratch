def _tracebacks(score_matrix, traceback_matrix, idx):
    """Implementation of traceeback.

    This version can produce empty tracebacks, which we generally don't want
    users seeing. So the higher level `tracebacks` filters those out.
    """
    score = score_matrix[idx]
    if score == 0:
        yield ()
        return

    directions = traceback_matrix[idx]

    assert directions != Direction.NONE, 'Tracebacks with direction NONE should have value 0!'

    row, col = idx

    if directions & Direction.UP.value:
        for tb in _tracebacks(score_matrix, traceback_matrix, (row - 1, col)):
            yield itertools.chain(tb, ((idx, Direction.UP),))

    if directions & Direction.LEFT.value:
        for tb in _tracebacks(score_matrix, traceback_matrix, (row, col - 1)):
            yield itertools.chain(tb, ((idx, Direction.LEFT),))

    if directions & Direction.DIAG.value:
        for tb in _tracebacks(score_matrix, traceback_matrix, (row - 1, col - 1)):
            yield itertools.chain(tb, ((idx, Direction.DIAG),))