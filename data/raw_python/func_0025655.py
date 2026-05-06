def decimal_date(dateobs, timeobs=None):
    """Convert DATE-OBS (and optional TIME-OBS) into a decimal year."""

    year, month, day = dateobs.split('-')
    if timeobs is not None:
        hr, min, sec = timeobs.split(':')
    else:
        hr, min, sec = 0, 0, 0

    rdate = datetime.datetime(int(year), int(month), int(day), int(hr),
                              int(min), int(sec))
    dday = (float(rdate.strftime("%j")) + rdate.hour / 24.0 +
            rdate.minute / (60. * 24) + rdate.second / (3600 * 24.)) / 365.25
    ddate = int(year) + dday

    return ddate