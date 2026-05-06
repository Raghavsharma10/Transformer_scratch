def from_jd(jd):
    '''Calculate Persian date from Julian day'''
    jd = trunc(jd) + 0.5

    depoch = jd - to_jd(475, 1, 1)
    cycle = trunc(depoch / 1029983)
    cyear = (depoch % 1029983)

    if cyear == 1029982:
        ycycle = 2820
    else:
        aux1 = trunc(cyear / 366)
        aux2 = cyear % 366
        ycycle = trunc(((2134 * aux1) + (2816 * aux2) + 2815) / 1028522) + aux1 + 1

    year = ycycle + (2820 * cycle) + 474

    if (year <= 0):
        year -= 1

    yday = (jd - to_jd(year, 1, 1)) + 1

    if yday <= 186:
        month = ceil(yday / 31)
    else:
        month = ceil((yday - 6) / 30)

    day = int(jd - to_jd(year, month, 1)) + 1

    return (year, month, day)