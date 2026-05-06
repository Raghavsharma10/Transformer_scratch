def easter(year):
    '''Calculate western easter'''
    # formula taken from http://aa.usno.navy.mil/faq/docs/easter.html
    c = trunc(year / 100)
    n = year - 19 * trunc(year / 19)

    k = trunc((c - 17) / 25)

    i = c - trunc(c / 4) - trunc((c - k) / 3) + (19 * n) + 15
    i = i - 30 * trunc(i / 30)
    i = i - trunc(i / 28) * (1 - trunc(i / 28) * trunc(29 / (i + 1)) * trunc((21 - n) / 11))

    j = year + trunc(year / 4) + i + 2 - c + trunc(c / 4)
    j = j - 7 * trunc(j / 7)

    l = i - j

    month = 3 + trunc((l + 40) / 44)
    day = l + 28 - 31 * trunc(month / 4)

    return year, int(month), int(day)