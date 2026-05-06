def to_jd(year, week, day):
    '''Return Julian day count of given ISO year, week, and day'''
    return day + n_weeks(SUN, gregorian.to_jd(year - 1, 12, 28), week)