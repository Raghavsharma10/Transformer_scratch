def to_jd(year, month, day):
    '''Determine Julian day from Persian date'''

    if year >= 0:
        y = 474
    else:
        y = 473
    epbase = year - y
    epyear = 474 + (epbase % 2820)

    if month <= 7:
        m = (month - 1) * 31
    else:
        m = (month - 1) * 30 + 6

    return day + m + trunc(((epyear * 682) - 110) / 2816) + (epyear - 1) * 365 + trunc(epbase / 2820) * 1029983 + (EPOCH - 1)