def filter_wait_on_queues(max_waiting):
    """Filter :class:`.Line` objects by their queueing time in
    HAProxy.

    :param max_waiting: maximum time, in milliseconds, a request is waiting on
      HAProxy prior to be delivered to a backend server. If HAProxy takes less
      than that time the log line is counted.
    :type max_waiting: string
    :returns: a function that filters by HAProxy queueing time.
    :rtype: function
    """
    def filter_func(log_line):
        waiting = int(max_waiting)
        return waiting >= log_line.time_wait_queues

    return filter_func