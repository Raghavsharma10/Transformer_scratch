def from_jd(jd):
    '''Convert a Julian day count to Positivist date.'''
    try:
        assert jd >= EPOCH
    except AssertionError:
        raise ValueError('Invalid Julian day')

    depoch = floor(jd - 0.5) + 0.5 - gregorian.EPOCH

    quadricent = floor(depoch / gregorian.INTERCALATION_CYCLE_DAYS)
    dqc = depoch % gregorian.INTERCALATION_CYCLE_DAYS

    cent = floor(dqc / gregorian.LEAP_SUPPRESSION_DAYS)
    dcent = dqc % gregorian.LEAP_SUPPRESSION_DAYS

    quad = floor(dcent / gregorian.LEAP_CYCLE_DAYS)
    dquad = dcent % gregorian.LEAP_CYCLE_DAYS

    yindex = floor(dquad / gregorian.YEAR_DAYS)
    year = (
        quadricent * gregorian.INTERCALATION_CYCLE_YEARS +
        cent * gregorian.LEAP_SUPPRESSION_YEARS +
        quad * gregorian.LEAP_CYCLE_YEARS + yindex
    )

    if yindex == 4:
        yearday = 365
        year = year - 1

    else:
        yearday = (
            depoch -
            quadricent * gregorian.INTERCALATION_CYCLE_DAYS -
            cent * gregorian.LEAP_SUPPRESSION_DAYS -
            quad * gregorian.LEAP_CYCLE_DAYS -
            yindex * gregorian.YEAR_DAYS
        )

    month = floor(yearday / 28)

    return (year - YEAR_EPOCH + 2, month + 1, int(yearday - (month * 28)) + 1)