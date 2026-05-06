def to_jd2(year, month, day):
    '''Gregorian to Julian Day Count for years between 1801-2099'''
    # http://quasar.as.utexas.edu/BillInfo/JulianDatesG.html

    legal_date(year, month, day)

    if month <= 2:
        year = year - 1
        month = month + 12

    a = floor(year / 100)
    b = floor(a / 4)
    c = 2 - a + b
    e = floor(365.25 * (year + 4716))
    f = floor(30.6001 * (month + 1))
    return c + day + e + f - 1524.5