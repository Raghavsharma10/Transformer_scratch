def check_error(result, func, cargs):
    "Error checking proper value returns"
    if result != 0:
        msg = 'Error in "%s": %s' % (func.__name__, get_errors(result) )
        raise RTreeError(msg)
    return