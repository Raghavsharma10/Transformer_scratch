def next_tzolkin(tzolkin, jd):
    '''For a given tzolk'in day, and a julian day count, find the next occurrance of that tzolk'in after the date'''
    if jd < EPOCH:
        raise IndexError("Input day is before Mayan epoch.")

    count1 = _tzolkin_count(*to_tzolkin(jd))
    count2 = _tzolkin_count(*tzolkin)

    add_days = (count2 - count1) % 260
    return jd + add_days