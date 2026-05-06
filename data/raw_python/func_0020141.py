def missing_intervals(startdate, enddate, start, end,
                      dateconverter=None,
                      parseinterval=None,
                      intervals=None):
    '''Given a ``startdate`` and an ``enddate`` dates, evaluate the
date intervals from which data is not available. It return a list of
two-dimensional tuples containing start and end date for the interval.
The list could countain 0,1 or 2 tuples.'''
    parseinterval = parseinterval or default_parse_interval
    dateconverter = dateconverter or todate
    startdate = dateconverter(parseinterval(startdate, 0))
    enddate = max(startdate, dateconverter(parseinterval(enddate, 0)))
    if intervals is not None and not isinstance(intervals, Intervals):
        intervals = Intervals(intervals)

    calc_intervals = Intervals()
    # we have some history already
    if start:
        # the startdate not available
        if startdate < start:
            calc_start = startdate
            calc_end = parseinterval(start, -1)
            if calc_end >= calc_start:
                calc_intervals.append(Interval(calc_start, calc_end))

        if enddate > end:
            calc_start = parseinterval(end, 1)
            calc_end = enddate
            if calc_end >= calc_start:
                calc_intervals.append(Interval(calc_start, calc_end))
    else:
        start = startdate
        end = enddate
        calc_intervals.append(Interval(startdate, enddate))

    if calc_intervals:
        if intervals:
            calc_intervals.extend(intervals)
    elif intervals:
        calc_intervals = intervals

    return calc_intervals