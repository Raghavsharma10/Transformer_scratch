def to_jd(year, month, day):
    '''Convert a Positivist date to Julian day count.'''
    legal_date(year, month, day)
    gyear = year + YEAR_EPOCH - 1

    return (
        gregorian.EPOCH - 1 + (365 * (gyear - 1)) +
        floor((gyear - 1) / 4) + (-floor((gyear - 1) / 100)) +
        floor((gyear - 1) / 400) + (month - 1) * 28 + day
    )