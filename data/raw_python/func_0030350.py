def strftime(dt, fmt):
    '''
    `strftime` implementation working before 1900
    '''
    if _illegal_s.search(fmt):
        raise TypeError("This strftime implementation does not handle %s")
    if dt.year > 1900:
        return dt.strftime(fmt)

    fmt = fmt.replace('%c', '%a %b %d %H:%M:%S %Y')\
             .replace('%Y', str(dt.year))\
             .replace('%y', '{:04}'.format(dt.year)[-2:])

    year = dt.year
    # For every non-leap year century, advance by
    # 6 years to get into the 28-year repeat cycle
    delta = 2000 - year
    off = 6*(delta // 100 + delta // 400)
    year = year + off

    # Move to around the year 2000
    year = year + ((2000 - year)//28)*28
    timetuple = dt.timetuple()
    return time.strftime(fmt, (year,) + timetuple[1:])