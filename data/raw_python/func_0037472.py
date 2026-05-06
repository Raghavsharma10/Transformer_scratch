def _query_wrap(fun, *args, **kwargs):
    """Wait until at least QUERY_WAIT_TIME seconds have passed
    since the last invocation of this function, then call the given
    function with the given arguments.
    """
    with _query_lock:
        global _last_query_time
        since_last_query = time.time() - _last_query_time
        if since_last_query < QUERY_WAIT_TIME:
            time.sleep(QUERY_WAIT_TIME - since_last_query)
        _last_query_time = time.time()
        return fun(*args, **kwargs)