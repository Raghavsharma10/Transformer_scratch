def next_haab(month, jd):
    '''For a given haab month and a julian day count, find the next start of that month on or after the JDC'''
    if jd < EPOCH:
        raise IndexError("Input day is before Mayan epoch.")

    hday, hmonth = to_haab(jd)

    if hmonth == month:
        days = 1 - hday

    else:
        count1 = _haab_count(hday, hmonth)
        count2 = _haab_count(1, month)

        # Find number of days between haab of given jd and desired haab
        days = (count2 - count1) % 365

    # add in the number of days and return new jd
    return jd + days