def to_jd(year, month, day):
    '''Determine Julian day from Bahai date'''
    gy = year - 1 + EPOCH_GREGORIAN_YEAR

    if month != 20:
        m = 0
    else:
        if isleap(gy + 1):
            m = -14
        else:
            m = -15
    return gregorian.to_jd(gy, 3, 20) + (19 * (month - 1)) + m + day