def localize_time_range(begin: datetime, end: datetime, tz=None) -> (datetime, datetime):
    """
    Localizes time range. Uses pytz.utc if None provided.
    :param begin: Begin datetime
    :param end: End datetime
    :param tz: pytz timezone or None (default UTC)
    :return: begin, end
    """
    if not tz:
        tz = pytz.utc
    return tz.localize(begin), tz.localize(end)