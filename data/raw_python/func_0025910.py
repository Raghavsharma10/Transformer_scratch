def _haab_count(day, month):
    '''Return the count of the given haab in the cycle. e.g. 0 Pop == 1, 5 Wayeb' == 365'''
    if day < 0 or day > 19:
        raise IndexError("Invalid day number")

    try:
        i = HAAB_MONTHS.index(month)
    except ValueError:
        raise ValueError("'{0}' is not a valid Haab' month".format(month))

    return min(i * 20, 360) + day