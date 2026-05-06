def normalize(seq):
    """
    Scales each number in the sequence so that the sum of all numbers equals 1.
    """
    s = float(sum(seq))
    return [v/s for v in seq]