def from_jd(jd):
    '''Return tuple of ISO (year, week, day) for Julian day'''
    year = gregorian.from_jd(jd)[0]
    day = jwday(jd) + 1

    dayofyear = ordinal.from_jd(jd)[1]
    week = trunc((dayofyear - day + 10) / 7)

    # Reset year
    if week < 1:
        week = weeks_per_year(year - 1)
        year = year - 1

    # Check that year actually has 53 weeks
    elif week == 53 and weeks_per_year(year) != 53:
        week = 1
        year = year + 1

    return year, week, day