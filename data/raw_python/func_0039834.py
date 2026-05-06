def retry_ex(callback, times=3, cap=120000):
    """
    Retry a callback function if any exception is raised.

    :param function callback: The function to call
    :keyword int times: Number of times to retry on initial failure
    :keyword int cap: Maximum wait time in milliseconds
    :returns: The return value of the callback
    :raises Exception: If the callback raises an exception after
      exhausting all retries
    """
    for attempt in range(times + 1):
        if attempt > 0:
            time.sleep(retry_wait_time(attempt, cap) / 1000.0)
        try:
            return callback()
        except:
            if attempt == times:
                raise