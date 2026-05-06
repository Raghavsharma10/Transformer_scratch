def add_month(t: datetime, n: int=1) -> datetime:
    """
    Adds +- n months to datetime.
    Clamps to number of days in given month.
    :param t: datetime
    :param n: count
    :return: datetime
    """
    t2 = t
    for count in range(abs(n)):
        if n > 0:
            t2 = datetime(year=t2.year, month=t2.month, day=1) + timedelta(days=32)
        else:
            t2 = datetime(year=t2.year, month=t2.month, day=1) - timedelta(days=2)
        try:
            t2 = t.replace(year=t2.year, month=t2.month)
        except Exception:
            first, last = monthrange(t2.year, t2.month)
            t2 = t.replace(year=t2.year, month=t2.month, day=last)
    return t2