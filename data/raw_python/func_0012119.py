def wait_any(futures, timeout=None):
    '''Wait for the completion of any (the first) one of multiple futures

    :param list futures: A list of :class:`Future`\s
    :param timeout:
        The maximum time to wait. With ``None``, will block indefinitely.
    :type timeout: float or None

    :returns:
        One of the futures from the provided list -- the first one to become
        complete (or any of the ones that were already complete).

    :raises WaitTimeout: if a timeout is provided and hit
    '''
    for fut in futures:
        if fut.complete:
            return fut

    wait = _Wait(futures)

    for fut in futures:
        fut._waits.add(wait)

    if wait.done.wait(timeout):
        raise errors.WaitTimeout()

    return wait.completed_future