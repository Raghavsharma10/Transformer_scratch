def get_last_day_of_month(t: datetime) -> int:
    """
    Returns day number of the last day of the month
    :param t: datetime
    :return: int
    """
    tn = t + timedelta(days=32)
    tn = datetime(year=tn.year, month=tn.month, day=1)
    tt = tn - timedelta(hours=1)
    return tt.day