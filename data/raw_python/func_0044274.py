def align(a, b, score_func, gap_penalty):
    """Calculate the best alignments of sequences `a` and `b`.

    Arguments:
      a: The first of two sequences to align
      b: The second of two sequences to align
      score_func: A 2-ary callable which calculates the "match" score between
        two elements in the sequences.
      gap_penalty: A 1-ary callable which calculates the gap penalty for a gap
        of a given size.

    Returns: A (score, alignments) tuple. `score` is the score that all of the
      `alignments` received. `alignments` is an iterable of `((index, index), .
      . .)` tuples describing the best (i.e. maximal and equally good)
      alignments. The first index in each pair is an index into `a` and the
      second is into `b`. Either (but not both) indices in a pair may be `None`
      indicating a gap in the corresponding sequence.

    """
    score_matrix, tb_matrix = build_score_matrix(a, b, score_func, gap_penalty)
    max_score = max(score_matrix.values())
    max_indices = (index
                   for index, score in score_matrix.items()
                   if score == max_score)
    alignments = (
        tuple(_traceback_to_alignment(tb, a, b))
        for idx in max_indices
        for tb in tracebacks(score_matrix, tb_matrix, idx))

    return (max_score, alignments)