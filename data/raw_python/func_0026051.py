def legal_date(year, month, day):
    '''Check if this is a legal date in the Gregorian calendar'''
    if month == 2:
        daysinmonth = 29 if isleap(year) else 28
    else:
        daysinmonth = 30 if month in HAVE_30_DAYS else 31

    if not (0 < day <= daysinmonth):
        raise ValueError("Month {} doesn't have a day {}".format(month, day))

    return True