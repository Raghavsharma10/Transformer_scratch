def argmin_list(seq, func):
    """ Return a list of elements of seq[i] with the lowest
        func(seq[i]) scores.
        >>> argmin_list(['one', 'to', 'three', 'or'], len)
        ['to', 'or']
    """
    best_score, best = func(seq[0]), []
    for x in seq:
        x_score = func(x)
        if x_score < best_score:
            best, best_score = [x], x_score
        elif x_score == best_score:
            best.append(x)
    return best