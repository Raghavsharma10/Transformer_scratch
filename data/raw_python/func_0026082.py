def to_jd(year, month, day, method=None):
    '''Obtain Julian day from a given French Revolutionary calendar date.'''
    method = method or 'equinox'

    if day < 1 or day > 30:
        raise ValueError("Invalid day for this calendar")

    if month > 13:
        raise ValueError("Invalid month for this calendar")

    if month == 13 and day > 5 + leap(year, method=method):
        raise ValueError("Invalid day for this month in this calendar")

    if method == 'equinox':
        return _to_jd_equinox(year, month, day)

    else:
        return _to_jd_schematic(year, month, day, method)