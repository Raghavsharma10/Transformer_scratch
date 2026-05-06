def dateJDN(year, month, day, calendar):
    """ Converts date to Julian Day Number. """
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12*a - 3
    if calendar == GREGORIAN:
        return day + (153*m + 2)//5 + 365*y + y//4 - y//100 + y//400 - 32045
    else:
        return day + (153*m + 2)//5 + 365*y + y//4 - 32083