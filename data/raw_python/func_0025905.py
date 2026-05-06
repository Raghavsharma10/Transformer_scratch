def from_jd(jd):
    '''Calculate Bahai date from Julian day'''

    jd = trunc(jd) + 0.5
    g = gregorian.from_jd(jd)
    gy = g[0]

    bstarty = EPOCH_GREGORIAN_YEAR

    if jd <= gregorian.to_jd(gy, 3, 20):
        x = 1
    else:
        x = 0
    # verify this next line...
    bys = gy - (bstarty + (((gregorian.to_jd(gy, 1, 1) <= jd) and x)))

    year = bys + 1
    days = jd - to_jd(year, 1, 1)
    bld = to_jd(year, 20, 1)

    if jd >= bld:
        month = 20
    else:
        month = trunc(days / 19) + 1
    day = int((jd + 1) - to_jd(year, month, 1))

    return year, month, day