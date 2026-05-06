def this_week(today: datetime=None, tz=None):
    """
    Returns this week begin (inclusive) and end (exclusive).
    :param today: Some date (defaults current datetime)
    :param tz: Timezone (defaults pytz UTC)
    :return: begin (inclusive), end (exclusive)
    """
    if today is None:
        today = datetime.utcnow()
    begin = today - timedelta(days=today.weekday())
    begin = datetime(year=begin.year, month=begin.month, day=begin.day)
    return localize_time_range(begin, begin + timedelta(days=7), tz)