def getHourTable(date, pos):
    """ Returns an HourTable object. """
    table = hourTable(date, pos)
    return HourTable(table, date)