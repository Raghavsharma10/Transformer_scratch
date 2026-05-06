def to_jd(year, month, day):
    '''Determine Julian day count from Islamic date'''
    return (day + ceil(29.5 * (month - 1)) + (year - 1) * 354 + trunc((3 + (11 * year)) / 30) + EPOCH) - 1