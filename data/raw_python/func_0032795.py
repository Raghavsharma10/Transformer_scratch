def semiyearly(date=datetime.date.today()):
    """
    Twice a year.
    """
    return datetime.date(date.year, 1 if date.month < 7 else 7, 1)