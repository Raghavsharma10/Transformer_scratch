def from_jd(jd):
    '''Calculate Islamic date from Julian day'''

    jd = trunc(jd) + 0.5
    year = trunc(((30 * (jd - EPOCH)) + 10646) / 10631)
    month = min(12, ceil((jd - (29 + to_jd(year, 1, 1))) / 29.5) + 1)
    day = int(jd - to_jd(year, month, 1)) + 1
    return (year, month, day)