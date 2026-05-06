def getLunarMonth11(yy, timeZone):
    '''def getLunarMonth11(yy, timeZone):  Find the day that starts the luner month
    11of the given year for the given time zone.'''
    # off = jdFromDate(31, 12, yy) \
    #            - 2415021.076998695
    off = jdFromDate(31, 12, yy) - 2415021.
    k = int(off / 29.530588853)
    nm = getNewMoonDay(k, timeZone)
    sunLong = getSunLongitude(nm, timeZone)
    # sun longitude at local midnight
    if (sunLong >= 9):
        nm = getNewMoonDay(k - 1, timeZone)
    return nm