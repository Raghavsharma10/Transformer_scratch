def next_tzolkin_haab(tzolkin, haab, jd):
    '''For a given haab-tzolk'in combination, and a Julian day count, find the next occurrance of the combination after the date'''
    # get H & T of input jd, and their place in the 18,980 day cycle
    haabcount = _haab_count(*to_haab(jd))
    haab_desired_count = _haab_count(*haab)

    # How many days between the input day and the desired day?
    haab_days = (haab_desired_count - haabcount) % 365

    possible_haab = set(h + haab_days for h in range(0, 18980, 365))

    tzcount = _tzolkin_count(*to_tzolkin(jd))
    tz_desired_count = _tzolkin_count(*tzolkin)
    # How many days between the input day and the desired day?
    tzolkin_days = (tz_desired_count - tzcount) % 260

    possible_tz = set(t + tzolkin_days for t in range(0, 18980, 260))

    try:
        return possible_tz.intersection(possible_haab).pop() + jd
    except KeyError:
        raise IndexError("That Haab'-Tzolk'in combination isn't possible")