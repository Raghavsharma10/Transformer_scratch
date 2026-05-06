def jdToDate(jd):
    '''def jdToDate(jd): Convert a Julian day number to day/month/year.
                       jd is an integer.'''
    if (jd > 2299160):
        # After 5/10/1582, Gregorian calendar
        a = jd + 32044
        b = int((4 * a + 3) / 146097.)
        c = a - int((b * 146097) / 4.)
    else:
        b = 0
        c = jd + 32082
    d = int((4 * c + 3) / 1461.)
    e = c - int((1461 * d) / 4.)
    m = int((5 * e + 2) / 153.)
    day = e - int((153 * m + 2) / 5.) + 1
    month = m + 3 - 12 * int(m / 10.)
    year = b * 100 + d - 4800 + int(m / 10.)
    return [day, month, year]