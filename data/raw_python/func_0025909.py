def to_tzolkin(jd):
    '''Determine Mayan Tzolkin "month" and day from Julian day'''
    lcount = trunc(jd) + 0.5 - EPOCH
    day = amod(lcount + 4, 13)
    name = amod(lcount + 20, 20)
    return int(day), TZOLKIN_NAMES[int(name) - 1]