def monthly(date=datetime.date.today()):
    """
    Take a date object and return the first day of the month.
    """
    return datetime.date(date.year, date.month, 1)