def to_haab(jd):
    '''Determine Mayan Haab "month" and day from Julian day'''
    # Number of days since the start of the long count
    lcount = trunc(jd) + 0.5 - EPOCH
    # Long Count begins 348 days after the start of the cycle
    day = (lcount + 348) % 365

    count = day % 20
    month = trunc(day / 20)

    return int(count), HAAB_MONTHS[month]