def getLeapMonthOffset(a11, timeZone):
    '''def getLeapMonthOffset(a11, timeZone): Find the index of the leap month
    after the month starting on the day a11.'''
    k = int((a11 - 2415021.076998695) / 29.530588853 + 0.5)
    last = 0
    i = 1  # start with month following lunar month 11
    arc = getSunLongitude(
        getNewMoonDay(k + i, timeZone), timeZone)
    while True:
        last = arc
        i += 1
        arc = getSunLongitude(
            getNewMoonDay(k + i, timeZone),
            timeZone)
        if not (arc != last and i < 14):
            break
    return i - 1