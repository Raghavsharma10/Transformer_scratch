def delay_1(year):
    '''Test for delay of start of new year and to avoid'''
    # Sunday, Wednesday, and Friday as start of the new year.
    months = trunc(((235 * year) - 234) / 19)
    parts = 12084 + (13753 * months)
    day = trunc((months * 29) + parts / 25920)

    if ((3 * (day + 1)) % 7) < 3:
        day += 1

    return day