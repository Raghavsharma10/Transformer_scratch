def mean_absolute_error(seq, correct):
    """
    Batch mean absolute error calculation.
    """
    assert len(seq) == len(correct)
    diffs = [abs(a-b) for a, b in zip(seq, correct)]
    return sum(diffs)/float(len(diffs))