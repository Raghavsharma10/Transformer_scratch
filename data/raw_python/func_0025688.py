def legal_date(year, month, day):
    '''Check if this is a legal date in the Julian calendar'''
    daysinmonth = month_length(year, month)

    if not (0 < day <= daysinmonth):
        raise ValueError("Month {} doesn't have a day {}".format(month, day))

    return True