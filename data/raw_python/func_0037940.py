def today_as_utc_datetime():
    """Datetime/Date comparisons aren't great, and someone might configure TODAY, to be a date."""
    now = today()
    if not isinstance(now, datetime) and isinstance(now, date):
        now = datetime.combine(now, datetime.min.time())
        now = now.replace(tzinfo=tz.gettz('UTC'))
    return now