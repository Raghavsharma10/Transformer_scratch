def from_jd(jd):
    '''Calculate Indian Civil date from Julian day
    Offset in years from Saka era to Gregorian epoch'''

    start = 80
    # Day offset between Saka and Gregorian

    jd = trunc(jd) + 0.5
    greg = gregorian.from_jd(jd)  # Gregorian date for Julian day
    leap = isleap(greg[0])  # Is this a leap year?
    # Tentative year in Saka era
    year = greg[0] - SAKA_EPOCH
    # JD at start of Gregorian year
    greg0 = gregorian.to_jd(greg[0], 1, 1)
    yday = jd - greg0  # Day number (0 based) in Gregorian year

    if leap:
        Caitra = 31  # Days in Caitra this year
    else:
        Caitra = 30

    if yday < start:
        # Day is at the end of the preceding Saka year
        year -= 1
        yday += Caitra + (31 * 5) + (30 * 3) + 10 + start

    yday -= start
    if yday < Caitra:
        month = 1
        day = yday + 1
    else:
        mday = yday - Caitra
        if (mday < (31 * 5)):
            month = trunc(mday / 31) + 2
            day = (mday % 31) + 1
        else:
            mday -= 31 * 5
            month = trunc(mday / 30) + 7
            day = (mday % 30) + 1

    return (year, month, int(day))