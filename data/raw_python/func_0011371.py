def FindFirst(params, ctxt, scope, stream, coord, interp):
    """
    This function is identical to the FindAll function except that the
    return value is the position of the first occurrence of the target
    found. A negative number is returned if the value could not be found.
    """
    global FIND_MATCHES_ITER
    FIND_MATCHES_ITER = _find_helper(params, ctxt, scope, stream, coord, interp)

    try:
        first = six.next(FIND_MATCHES_ITER)
        return first.start() + FIND_MATCHES_START_OFFSET
    except StopIteration as e:
        return -1