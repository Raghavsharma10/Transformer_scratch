def FindNext(params, ctxt, scope, stream, coord):
    """
    This function returns the position of the next occurrence of the
    target value specified with the FindFirst function. If dir is 1, the
    find direction is down. If dir is 0, the find direction is up. The
    return value is the address of the found data, or -1 if the target
    is not found.
    """
    if FIND_MATCHES_ITER is None:
        raise errors.InvalidState()
    
    direction = 1
    if len(params) > 0:
        direction = PYVAL(params[0])
    
    if direction != 1:
        # TODO maybe instead of storing the iterator in FIND_MATCHES_ITER,
        # we should go ahead and find _all the matches in the file and store them
        # in a list, keeping track of the idx of the current match.
        #
        # This would be highly inefficient on large files though.
        raise NotImplementedError("Reverse searching is not yet implemented")
    
    try:
        next_match = six.next(FIND_MATCHES_ITER)
        return next_match.start() + FIND_MATCHES_START_OFFSET
    except StopIteration as e:
        return -1