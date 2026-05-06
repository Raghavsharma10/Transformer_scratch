def independence_day(year, observed=None):
    '''July 4th'''
    day = 4

    if observed:
        if calendar.weekday(year, JUL, 4) == SAT:
            day = 3

        if calendar.weekday(year, JUL, 4) == SUN:
            day = 5

    return (year, JUL, day)