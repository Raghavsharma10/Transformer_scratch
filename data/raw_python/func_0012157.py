def elapsed_time_string(start_time, stop_time):
    r"""
    Return a formatted string with the elapsed time between two time points.

    The string includes years (365 days), months (30 days), days (24 hours),
    hours (60 minutes), minutes (60 seconds) and seconds. If both arguments
    are equal, the string returned is :code:`'None'`; otherwise, the string
    returned is [YY year[s], [MM month[s], [DD day[s], [HH hour[s],
    [MM minute[s] [and SS second[s\]\]\]\]\]\]. Any part (year[s], month[s],
    etc.) is omitted if the value of that part is null/zero

    :param start_time: Starting time point
    :type  start_time: `datetime <https://docs.python.org/3/library/
                       datetime.html#datetime-objects>`_

    :param stop_time: Ending time point
    :type  stop_time: `datetime`

    :rtype: string

    :raises: RuntimeError (Invalid time delta specification)

    For example:

        >>> import datetime, pmisc
        >>> start_time = datetime.datetime(2014, 1, 1, 1, 10, 1)
        >>> stop_time = datetime.datetime(2015, 1, 3, 1, 10, 3)
        >>> pmisc.elapsed_time_string(start_time, stop_time)
        '1 year, 2 days and 2 seconds'
    """
    if start_time > stop_time:
        raise RuntimeError("Invalid time delta specification")
    delta_time = stop_time - start_time
    # Python 2.6 datetime objects do not have total_seconds() method
    tot_seconds = int(
        (
            delta_time.microseconds
            + (delta_time.seconds + delta_time.days * 24 * 3600) * 10 ** 6
        )
        / 10 ** 6
    )
    years, remainder = divmod(tot_seconds, 365 * 24 * 60 * 60)
    months, remainder = divmod(remainder, 30 * 24 * 60 * 60)
    days, remainder = divmod(remainder, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    token_iter = zip(
        [years, months, days, hours, minutes, seconds],
        ["year", "month", "day", "hour", "minute", "second"],
    )
    ret_list = [
        "{token} {token_name}{plural}".format(
            token=num, token_name=desc, plural="s" if num > 1 else ""
        )
        for num, desc in token_iter
        if num > 0
    ]
    if not ret_list:
        return "None"
    if len(ret_list) == 1:
        return ret_list[0]
    if len(ret_list) == 2:
        return ret_list[0] + " and " + ret_list[1]
    return (", ".join(ret_list[0:-1])) + " and " + ret_list[-1]