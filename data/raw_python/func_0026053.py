def from_jd(jd):
    '''Return Gregorian date in a (Y, M, D) tuple'''
    wjd = floor(jd - 0.5) + 0.5
    depoch = wjd - EPOCH

    quadricent = floor(depoch / INTERCALATION_CYCLE_DAYS)
    dqc = depoch % INTERCALATION_CYCLE_DAYS

    cent = floor(dqc / LEAP_SUPPRESSION_DAYS)
    dcent = dqc % LEAP_SUPPRESSION_DAYS

    quad = floor(dcent / LEAP_CYCLE_DAYS)
    dquad = dcent % LEAP_CYCLE_DAYS

    yindex = floor(dquad / YEAR_DAYS)
    year = (
        quadricent * INTERCALATION_CYCLE_YEARS +
        cent * LEAP_SUPPRESSION_YEARS +
        quad * LEAP_CYCLE_YEARS + yindex
    )

    if not (cent == 4 or yindex == 4):
        year += 1

    yearday = wjd - to_jd(year, 1, 1)

    leap = isleap(year)

    if yearday < 58 + leap:
        leap_adj = 0
    elif leap:
        leap_adj = 1
    else:
        leap_adj = 2

    month = floor((((yearday + leap_adj) * 12) + 373) / 367)
    day = int(wjd - to_jd(year, month, 1)) + 1

    return (year, month, day)