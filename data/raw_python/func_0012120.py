def wait_all(futures, timeout=None):
    '''Wait for the completion of all futures in a list

    :param list future: a list of :class:`Future`\s
    :param timeout:
        The maximum time to wait. With ``None``, can block indefinitely.
    :type timeout: float or None

    :raises WaitTimeout: if a timeout is provided and hit
    '''
    if timeout is not None:
        deadline = time.time() + timeout
        for fut in futures:
            fut.wait(deadline - time.time())
    else:
        for fut in futures:
            fut.wait()