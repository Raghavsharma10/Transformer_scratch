def convertDate(date):
    """Convert DATE string into a decimal year."""

    d, t = date.split('T')
    return decimal_date(d, timeobs=t)