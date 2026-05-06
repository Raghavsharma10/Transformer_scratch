def biweekly(date=datetime.date.today()):
    """
    Every two weeks.
    """
    return datetime.date(date.year, date.month, 1 if date.day < 15 else 15)