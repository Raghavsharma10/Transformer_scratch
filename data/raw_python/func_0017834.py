def _retry_func(func, param, num, retry_notif, error_msg):
    """
    A function which retries a given function num times and calls retry_notif each
    time the function is retried.
    :param func: The function to retry num times.
    :param num: The number of times to try before giving up.
    :param retry_notif: Will be called with the same parameter as func if we have to retry the
                        function. Will also receive the number of retries so far as a second
                        parameter.
    :param: error_msg: The message

    Throws DatacatsError if we run out of retries. Returns otherwise.
    """
    for retry_num in range(num):
        if retry_num:
            retry_notif(param, retry_num)
        try:
            func(param)
            return
        except DatacatsError:
            pass

    raise DatacatsError(error_msg)