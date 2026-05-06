def quarterly(date=datetime.date.today()):
    """
    Fixed at: 1/1, 4/1, 7/1, 10/1.
    """
    return datetime.date(date.year, ((date.month - 1)//3) * 3 + 1, 1)