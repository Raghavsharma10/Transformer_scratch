def stat_holidays(province='BC', year=2015):
    """ Returns a list of holiday dates for a province and year. """
    return holidays.Canada(state=province, years=year).keys()