def dayname(year, month, day):
    '''
    Give the name of the month and day for a given date.

    Returns:
        tuple month_name, day_name
    '''
    legal_date(year, month, day)

    yearday = (month - 1) * 28 + day

    if isleap(year + YEAR_EPOCH - 1):
        dname = data.day_names_leap[yearday - 1]
    else:
        dname = data.day_names[yearday - 1]

    return MONTHS[month - 1], dname