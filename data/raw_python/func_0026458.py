def toYearFraction(date):
    """Converts :class:`datetime.date` or :class:`datetime.datetime` to decimal
    year.

    Parameters
    ==========
    date : :class:`datetime.date` or :class:`datetime.datetime`

    Returns
    =======
    year : float
        Decimal year

    Notes
    =====
    The algorithm is taken from http://stackoverflow.com/a/6451892/2978652

    """

    def sinceEpoch(date):
        """returns seconds since epoch"""
        return time.mktime(date.timetuple())
    year = date.year
    startOfThisYear = dt.datetime(year=year, month=1, day=1)
    startOfNextYear = dt.datetime(year=year+1, month=1, day=1)

    yearElapsed = sinceEpoch(date) - sinceEpoch(startOfThisYear)
    yearDuration = sinceEpoch(startOfNextYear) - sinceEpoch(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction