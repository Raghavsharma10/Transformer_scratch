def build_score_matrix(a, b, score_func, gap_penalty):
    """Calculate the score and traceback matrices for two input sequences and
    scoring functions.

    Returns: A tuple of (score-matrix, traceback-matrix). Each entry in the
      score-matrix is a numeric score. Each entry in the traceback-matrix is a
      logical ORing of the direction bitfields.
    """
    score_matrix = Matrix(rows=len(a) + 1, cols=len(b) + 1)
    traceback_matrix = Matrix(rows=len(a) + 1, cols=len(b) + 1, type_code='B')

    for row in range(1, score_matrix.rows):
        for col in range(1, score_matrix.cols):
            match_score = score_func(a[row - 1], b[col - 1])

            scores = sorted(
                ((score_matrix[(row - 1, col - 1)] + match_score,
                  Direction.DIAG),
                 (score_matrix[(row - 1, col)] - gap_penalty(1),
                  Direction.UP),
                 (score_matrix[(row, col - 1)] - gap_penalty(1),
                  Direction.LEFT),
                 (0, Direction.NONE)),
                key=lambda x: x[0],
                reverse=True)
            max_score = scores[0][0]
            scores = itertools.takewhile(
                lambda x: x[0] == max_score,
                scores)

            score_matrix[row, col] = max_score
            for _, direction in scores:
                traceback_matrix[row, col] = traceback_matrix[row, col] | direction.value

    return score_matrix, traceback_matrix