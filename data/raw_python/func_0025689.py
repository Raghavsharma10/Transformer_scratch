def from_jd(jd):
    '''Calculate Julian calendar date from Julian day'''

    jd += 0.5
    z = trunc(jd)

    a = z
    b = a + 1524
    c = trunc((b - 122.1) / 365.25)
    d = trunc(365.25 * c)
    e = trunc((b - d) / 30.6001)

    if trunc(e < 14):
        month = e - 1
    else:
        month = e - 13

    if trunc(month > 2):
        year = c - 4716
    else:
        year = c - 4715

    day = b - d - trunc(30.6001 * e)

    return (year, month, day)