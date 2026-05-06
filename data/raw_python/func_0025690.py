def to_jd(year, month, day):
    '''Convert to Julian day using astronomical years (0 = 1 BC, -1 = 2 BC)'''

    legal_date(year, month, day)

    # Algorithm as given in Meeus, Astronomical Algorithms, Chapter 7, page 61

    if month <= 2:
        year -= 1
        month += 12

    return (trunc((365.25 * (year + 4716))) + trunc((30.6001 * (month + 1))) + day) - 1524.5