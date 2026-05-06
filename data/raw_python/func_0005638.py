def this_month(today: datetime=None, tz=None):
    """
    Returns current month begin (inclusive) and end (exclusive).
    :param today: Some date in the month (defaults current datetime)
    :param tz: Timezone (defaults pytz UTC)
    :return: begin (inclusive), end (exclusive)
    """
    if today is None:
        today = datetime.utcnow()
    begin = datetime(day=1, month=today.month, year=today.year)
    end = begin + timedelta(days=32)
    end = datetime(day=1, month=end.month, year=end.year)
    return localize_time_range(begin, end, tz)