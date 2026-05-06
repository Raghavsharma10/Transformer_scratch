def date_to_string(date):
    """Transform a date or datetime object into a string and return it.

    Examples:
    >>> date_to_string(datetime.datetime(2012, 1, 3, 12, 23, 34, tzinfo=UTC))
    '2012-01-03T12:23:34+00:00'
    >>> date_to_string(datetime.datetime(2012, 1, 3, 12, 23, 34))
    '2012-01-03T12:23:34'
    >>> date_to_string(datetime.date(2012, 1, 3))
    '2012-01-03'

    """
    if isinstance(date, datetime.datetime):
        # Create an ISO 8601 datetime string
        date_str = date.strftime('%Y-%m-%dT%H:%M:%S')
        tzstr = date.strftime('%z')
        if tzstr:
            # Yes, this is ugly. And no, I haven't found a better way to have a
            # truly ISO 8601 datetime with timezone in Python.
            date_str = '%s%s:%s' % (date_str, tzstr[0:3], tzstr[3:5])
    elif isinstance(date, datetime.date):
        # Create an ISO 8601 date string
        date_str = date.strftime('%Y-%m-%d')
    else:
        raise TypeError('Argument is not a date or datetime. ')

    return date_str