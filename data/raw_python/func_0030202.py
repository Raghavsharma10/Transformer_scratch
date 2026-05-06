def skipping_window(sequence, target, n=3):
    """
    Return a sliding window with a constraint to check that
    target is inside the window.
    From http://stackoverflow.com/q/43626525/610569

        >>> list(skipping_window([1,2,3,4,5], 2, 3))
        [(1, 2, 3), (2, 3, 4)]
    """
    start, stop = 0, n
    seq = list(sequence)
    while stop <= len(seq):
        subseq = seq[start:stop]
        if target in subseq:
            yield tuple(seq[start:stop])
        start += 1
        stop += 1
        # Fast forwarding the start.
        # Find the next window which contains the target.
        try:
            # `seq.index(target, start) - (n-1)` would be the next
            # window where the constraint is met.
            start = max(seq.index(target, start) - (n-1), start)
            stop = start + n
        except ValueError:
            break