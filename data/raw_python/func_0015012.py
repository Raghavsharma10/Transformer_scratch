def filter_slow_requests(slowness):
    """Filter :class:`.Line` objects by their response time.

    :param slowness: minimum time, in milliseconds, a server needs to answer
      a request. If the server takes more time than that the log line is
      accepted.
    :type slowness: string
    :returns: a function that filters by the server response time.
    :rtype: function
    """
    def filter_func(log_line):
        slowness_int = int(slowness)
        return slowness_int <= log_line.time_wait_response

    return filter_func