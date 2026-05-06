def filter_time_frame(start, delta):
    """Filter :class:`.Line` objects by their connection time.

    :param start: a time expression (see -s argument on --help for its format)
      to filter log lines that are before this time.
    :type start: string
    :param delta: a relative time expression (see -s argument on --help for
      its format) to limit the amount of time log lines will be considered.
    :type delta: string
    :returns: a function that filters by the time a request is made.
    :rtype: function
    """
    start_value = start
    delta_value = delta
    end_value = None

    if start_value is not '':
        start_value = _date_str_to_datetime(start_value)

    if delta_value is not '':
        delta_value = _delta_str_to_timedelta(delta_value)

    if start_value is not '' and delta_value is not '':
        end_value = start_value + delta_value

    def filter_func(log_line):
        if start_value is '':
            return True
        elif start_value > log_line.accept_date:
            return False

        if end_value is None:
            return True
        elif end_value < log_line.accept_date:
            return False

        return True

    return filter_func