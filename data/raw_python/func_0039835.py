def retry_bool(callback, times=3, cap=120000):
    """
    Retry a callback function if it returns False.

    :param function callback: The function to call
    :keyword int times: Number of times to retry on initial failure
    :keyword int cap: Maximum wait time in milliseconds
    :returns: The return value of the callback
    :rtype: bool
    """
    for attempt in range(times + 1):
        if attempt > 0:
            time.sleep(retry_wait_time(attempt, cap) / 1000.0)
        ret = callback()
        if ret or attempt == times:
            break
    return ret