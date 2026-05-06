def _block(predicate, timeout):
    """
    Block until a predicate becomes true.

    ``predicate`` is a function taking no arguments. The call to
    ``_block`` blocks until ``predicate`` returns a true value. This
    is done by polling ``predicate``.

    ``timeout`` is either ``True`` (block indefinitely) or a timeout
    in seconds.

    The return value is the value of the predicate after the
    timeout.
    """
    if timeout:
        if timeout is True:
            timeout = float('Inf')
        timeout = time.time() + timeout
        while not predicate() and time.time() < timeout:
            time.sleep(0.1)
    return predicate()