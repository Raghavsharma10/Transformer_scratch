def parse_datetime_range(time_filter):
    """
    Parse the url param to python objects.
    From what time range to divide by a.time.gap into intervals.
    Defaults to q.time and otherwise 90 days.
    Validate in API: re.search("\\[(.*) TO (.*)\\]", value)
    :param time_filter: [2013-03-01 TO 2013-05-01T00:00:00]
    :return: datetime.datetime(2013, 3, 1, 0, 0), datetime.datetime(2013, 5, 1, 0, 0)
    """

    if not time_filter:
        time_filter = "[* TO *]"

    start, end = parse_solr_time_range_as_pair(time_filter)
    start, end = parse_datetime(start), parse_datetime(end)
    return start, end