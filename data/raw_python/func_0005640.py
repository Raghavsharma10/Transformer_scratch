def last_year(today: datetime=None, tz=None):
    """
    Returns last year begin (inclusive) and end (exclusive).
    :param today: Some date (defaults current datetime)
    :param tz: Timezone (defaults pytz UTC)
    :return: begin (inclusive), end (exclusive)
    """
    if today is None:
        today = datetime.utcnow()
    end = datetime(day=1, month=1, year=today.year)
    end_incl = end - timedelta(seconds=1)
    begin = datetime(day=1, month=1, year=end_incl.year)
    return localize_time_range(begin, end, tz)